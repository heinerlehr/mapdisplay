import os
import csv
from datetime import datetime
from functools import lru_cache
from io import BytesIO
import base64

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pathlib import Path
import param
import panel as pn
from cryptography.fernet import Fernet
import folium

# FastAPI
from fastapi import FastAPI
from fastapi.responses import Response
from panel.io.fastapi import add_application
from contextlib import asynccontextmanager

# Configuration
from iconfig.iconfig import iConfig

from loguru import logger

from md.model.version import Version
from md.storage.storage import create_storage
from md.model.models import TileAddress


config = iConfig()


# Create FastAPI app
app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("FastAPI server starting...")
    logger.info("Panel app will be available at /panel")
    try:
        # Pre-initialize the app container to catch any errors early
        container = get_app_container()
        logger.info(f"App container initialized. Authenticated: {container.authenticated}")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    yield
    
    # Shutdown
    logger.info("FastAPI server shutting down...")

app = FastAPI(lifespan=lifespan)

# Health check / root endpoint
@app.get("/")
async def root():
    """Root endpoint - redirect to Panel app."""
    from fastapi.responses import HTMLResponse
    return HTMLResponse("""
        <html>
            <head>
                <title>Map Visualizer</title>
                <meta http-equiv="refresh" content="0; url=/panel" />
            </head>
            <body>
                <h1>Map Visualizer</h1>
                <p>Redirecting to <a href="/panel">/panel</a>...</p>
            </body>
        </html>
    """)


# In-memory tile cache (LRU cache with max 1000 tiles)
@lru_cache(maxsize=1000)
def _load_tile_cached(version: str, var: str, species: str, time: str, z: int, x: int, y: int) -> bytes:
    """Load and cache tile bytes in memory."""
    ta = TileAddress(version=version, var=var, species=species, time=time, z=z, x=x, y=y)
    storage = create_storage()
    return storage.load_tile(ta)

#################################################################################################
#
# TILE SERVING
#
#################################################################################################

@app.get("/tile/{version}/{var}/{species}/{time}/{z}/{x}/{y}")
async def get_tile(version: str, var: str, species: str, time: str, z: int, x: int, y: int):
    """Serve tiles for the map with caching.
    
    Includes:
    - In-memory LRU cache (1000 most recent tiles)
    - HTTP cache headers (immutable tiles can be cached aggressively)
    """
    try:
        # Load from cache (or blob storage if not cached)
        tile_bytes = _load_tile_cached(version=version, var=var, species=species, time=time, z=z, x=x, y=y)
        
        # Return with caching headers
        # During development, disable caching so tile updates appear immediately
        # In production, enable aggressive caching since tiles are immutable
        return Response(
            content=tile_bytes,
            media_type="image/png",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",  # Disable caching for tile updates
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )
    except Exception as e:
        logger.error(f"Failed to load tile {var}/{species}/{time}/{z}/{x}/{y}: {e}")
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": str(e)}, status_code=404)


#################################################################################################
#
# AUTHENTICATION
#
#################################################################################################

class AuthManager:
    """Simple authentication manager for multi-user sessions."""
    
    def __init__(self):
        self.sessions = {}  # Maps session_id -> username

    def _decrypt_password(self, encrypted_password):
        key = self._load_or_create_key()
        cipher = Fernet(key)
        return cipher.decrypt(encrypted_password.encode()).decode()

    def _retrieve_credentials(self)->dict:
        credentials = {}
        filename = Path(os.getenv("CREDENTIALS_FILE", default="credentials.csv"))
        with open(filename, mode="r", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                username, encrypted_password = row
                decrypted_password = self._decrypt_password(encrypted_password)
                credentials[username] = decrypted_password
        return credentials

    def _load_or_create_key(self):
        key_fn = Path(os.getenv("SECRETS_FILE", default="secret.key"))
        """Loads existing key or creates a new one."""
        if key_fn.exists():
            with open(key_fn, "rb") as key_file:
                return key_file.read()
        else:
            logger.error(f"No secrets file {key_fn} found.")
            raise FileNotFoundError(f"No secrets file {key_fn} found.")    
    
    def authenticate(self, username: str, password: str) -> bool:
        """Verify username and password."""
        credentials = self._retrieve_credentials()
        return credentials.get(username) == password
    
    def create_session(self, username: str) -> str:
        """Create a session for authenticated user."""
        import uuid
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {"username": username, "created_at": datetime.now()}
        logger.info(f"Session created for user: {username}")
        return session_id
    
    def get_username(self, session_id: str) -> str:
        """Get username from session ID."""
        return self.sessions.get(session_id, {}).get("username", "anonymous")
    
    def is_authenticated(self, session_id: str) -> bool:
        """Check if session is valid."""
        return session_id in self.sessions


auth_manager = AuthManager()

#################################################################################################
#
# PANEL APP - PARAMETERIZED CLASS
#
#################################################################################################

class MapVisualizerApp(param.Parameterized):
    """Interactive map visualization app with metadata-driven controls."""
    
    # Metadata - loaded on initialization
    metadata = param.Dict(default={}, precedence=-1)
    username = param.String(default="anonymous", precedence=-1)
    on_logout = param.Callable(default=None, precedence=-1)  # Callback to parent container
    
    # User controls
    version = param.Selector(default=None, objects=[])  # Version selector (if multiple versions are supported in the future)
    variable = param.Selector(default='suitability', objects=[])
    species = param.Selector(default=None, objects=[])
    use_mean = param.Boolean(default=True, precedence=1)  # Checkbox: use mean vs specific time
    time_step_index = param.Integer(default=0, bounds=(0, 0), precedence=1)  # Slider for specific timestep
    show_mask = param.Boolean(default=False)
    
    # Preserve map state across updates
    zoom_level = param.Integer(default=2, precedence=-1)
    
    # Internal mapping of slider index to time parameter (for URL generation)
    _time_index_map = {}  # Maps {slider_index: "mean" or timestep_index}
    
    def __init__(self, username: str = "anonymous", on_logout=None, **params):
        self.username = username
        self.on_logout = on_logout
        super().__init__(**params)
        versions = Version.load_versions()
        self.param.version.objects = [version.id for version in versions]
        if versions:
            self.version = versions[0].id

        self._load_metadata()
    
    def _load_metadata(self):
        """Fetch metadata directly from storage."""
        try:
            storage = create_storage()
            map_definition = storage.load_map_definition(version_id=self.version)
            
            if map_definition is None:
                logger.warning(f"Metadata not found for version {self.version}")
                self.map_definition = {}
                return

            # Populate parameter choices from metadata
            if "variables" not in map_definition:
                raise ValueError("Metadata is missing 'variables' key")
            
            self.variable_configs = map_definition.variables
            varnames = [var.name for var in self.variable_configs if not var.is_mask]
            self.param.variable.objects = varnames
            self.variable = varnames[0]

            # Set min max for the color bar
            self.vmin = self.variable_configs[0].vmin
            self.vmax = self.variable_configs[0].vmax
            
            if "species" in map_definition:
                self.param.species.objects = map_definition.get("species", [])
                if map_definition.get("species"):
                    self.species = map_definition["species"][0]
            
            # Build time slider: map slider position (0=mean, 1+=timesteps) to time index
            self._time_index_map = {}
            
            if "time_labels" in map_definition and "time_range" in map_definition:
                time_labels = map_definition.get("time_labels", [])
                time_range = map_definition.get("time_range", [])
                for idx, label in zip(time_range, time_labels):
                    self._time_index_map[str(idx)] = label
            elif "time_labels" in map_definition:
                time_labels = map_definition.get("time_labels", [])
                for idx, label in enumerate(time_labels):
                    self._time_index_map[str(idx)] = label

            elif "time_range" in map_definition:
                time_range = map_definition.get("time_range", [])
                for idx in time_range:
                    self._time_index_map[str(idx)] = str(idx)
            
            logger.info(f"Metadata loaded successfully for user {self.username}: {self.map_definition}")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.map_definition = {}
    
    @param.depends("variable", "species", "use_mean", "time_step_index", "show_mask", watch=True)
    def _on_params_changed(self):
        """Update visualization when parameters change."""
        time_value = "mean" if self.use_mean else str(self.time_step_index)
        logger.info(f"[{self.username}] Parameters changed: var={self.variable}, species={self.species}, time={time_value}, mask={self.show_mask}")
    
    @pn.depends("variable")
    def _generate_colorbar(self):
        """Generate a colorbar image for the current variable."""
        try:
            variable_config = next((var for var in self.variable_configs if var.name == self.variable), None)
            if variable_config is None:
                logger.warning(f"No variable config found for {self.variable}")
                return pn.pane.Markdown("*No colorbar available*")
            # Get colormap for current variable (simple mapping, could be enhanced)
            colormap_name = variable_config.colormap or "viridis"

            vmin = variable_config.vmin
            vmax = variable_config.vmax
            
            # Create a simple colorbar image
            fig, ax = plt.subplots(figsize=(1.5, 4), dpi=100)
            cmap = cm.get_cmap(colormap_name)
            
            # Create gradient data: 0 to 1 on y-axis
            gradient = np.linspace(vmin, vmax, 256).reshape(256, 1)
            
            # Display gradient
            ax.imshow(gradient, aspect='auto', cmap=cmap, origin='lower', extent=[0, 1, vmin, vmax])
            ax.set_xticks([])
            ax.set_ylabel('Value', fontsize=14)
            ax.tick_params(axis='y', labelsize=12)
            
            plt.tight_layout()
            
            # Convert to base64 for embedding in HTML
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            html = f'<img src="data:image/png;base64,{img_base64}" width="100" style="margin: 10px 0;"/>'
            return pn.pane.HTML(html)
        except Exception as e:
            logger.warning(f"Failed to generate colorbar: {e}")
            return pn.pane.Markdown("*Colorbar unavailable*")
    
    @pn.depends("variable", "species", "use_mean", "time_step_index")
    def create_map(self):
        """Create a Folium map with tiles from the server."""
        try:
            # Get the actual time value: use mean or specific timestep
            time_value = "mean" if self.use_mean else str(self.time_step_index)
            
            # Create base map centered on a default location, preserving zoom level
            m = folium.Map(
                location=[20, 0],
                zoom_start=self.zoom_level,
                tiles='Esri.OceanBasemap',
            )
            
            # Add custom tile layer pointing to our tile server
            tile_url = f"/tile/{self.version}/{self.variable}/{self.species}/{time_value}/{{z}}/{{x}}/{{y}}"
            folium.TileLayer(
                tiles=tile_url,
                attr="Map Visualizer",
                name=f"{self.variable} - {self.species}",
                overlay=True,
                control=True,
                max_zoom=self._max_zoom_level,
                opacity=1.0,  # Full opacity to allow PNG alpha transparency to show through
                tms=False,  # TMS coordinate scheme (y flipped) since our tiles are in TMS format
            ).add_to(m)

            # Add custom tile layer pointing to our tile server
            mask_url = f"/tile/{self.version}/mask/{self.species}/{time_value}/{{z}}/{{x}}/{{y}}"
            folium.TileLayer(
                tiles=mask_url,
                attr="Mask",
                name=f"Mask - {self.species}",
                overlay=True,
                control=True,
                max_zoom=self._max_zoom_level,
                opacity=1.0,  # Full opacity to allow PNG alpha transparency to show through
                tms=False,  # TMS coordinate scheme (y flipped) since our tiles are in TMS format
            ).add_to(m)
            
            folium.LayerControl().add_to(m)
            
            # Create an HTML pane that captures zoom level changes
            map_html = m._repr_html_()
            
            # Use a custom callback to capture zoom level from folium's internal state
            # This is a workaround since folium doesn't expose zoom_level directly in _repr_html_
            return pn.pane.HTML(map_html, sizing_mode='stretch_width', height=1024)
        except Exception as e:
            logger.error(f"Failed to create map: {e}")
            return pn.pane.Markdown(f"❌ Error creating map: {str(e)}")
    
    def panel(self):
        """Create the main panel layout."""
        
        def on_logout_click(event):
            """Handle logout button click."""
            if self.on_logout:
                self.on_logout()
        
        logout_button = pn.widgets.Button(name="Logout", button_type="warning")
        logout_button.on_click(on_logout_click)
        
        # Create time controls: checkbox for mean + date selector
        mean_checkbox = pn.widgets.Checkbox(name="Use Mean", value=True)
        mean_checkbox.link(self, value='use_mean')
        
        # Create time selector for specific timesteps (shows actual dates from metadata)
        # Panel Select: {display_label: return_value}, so swap the dict
        time_values = [pd.Timestamp(ts).to_pydatetime() for ts in self._time_index_map.values()]
        time_selector = pn.widgets.DateSlider(
            name="Date",
            start=min(time_values),
            end = max(time_values),
            value=min(time_values),
        )

        # Update bounds of time_step_index based on available time steps in metadata
        min_index = int(list(self._time_index_map.keys())[0])
        max_index = int(list(self._time_index_map.keys())[-1])
        self.param.time_step_index.bounds = (min_index, max_index)
        self.param.time_step_index.default = min_index
        self.time_step_index = min_index

        self._max_zoom_level = config("tiler.max_zoom_levels", default=2)
        
        # Watch selector changes from user and update time_step_index
        # (unidirectional only to avoid circular triggers)
        def on_time_selector_change(event):
            try:
                # event.new is the key from time_options dict
                time_values = pd.Series([pd.Timestamp(ts) for ts in self._time_index_map.values()])
                current = pd.Timestamp(event.new)
                arg = (time_values-current).abs().argmin()  # Find closest time index
                self.time_step_index = int(list(self._time_index_map.keys())[arg])
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid time selector value: {event.new} - {e}")
        
        time_selector.param.watch(on_time_selector_change, 'value_throttled', onlychanged=True)
        
        # Disable selector when use_mean is True
        def update_selector_disabled(*events):
            time_selector.disabled = self.use_mean
        
        self.param.watch(update_selector_disabled, 'use_mean')
        time_selector.disabled = self.use_mean  # Initialize disabled state
        
        # Stack: checkbox + selector
        time_controls = pn.Column(
            mean_checkbox,
            time_selector,
            sizing_mode='stretch_width'
        )
        
        controls = pn.Column(
            pn.pane.Markdown("## Map Controls", margin=(0, 10)),
            self.param.version,
            self.param.variable,
            self.param.species,
            time_controls,
            pn.layout.Divider(),
            self._generate_colorbar,  # Add colorbar here
            width=300,
            sizing_mode='fixed',
        )
        
        # Create header row with username on left and logout button on right
        logout_button.width = 90
        header_row = pn.Row(
            pn.pane.Markdown(f"### Logged in as: {self.username}", margin=(10, 10)),
            pn.layout.Spacer(),  # Flexible space to push button to right
            logout_button,
            margin=(0, 10),
        )
        
        # Return layout with header, controls sidebar, and map
        # Pass create_map method directly (not the result) so Panel tracks parameter changes
        return pn.Column(
            header_row,
            pn.Row(controls, self.create_map, sizing_mode='stretch_both'),
            width=1800,
            height=1024,
        )


#################################################################################################
#
# LOGIN PAGE & APP WRAPPER
#
#################################################################################################

class AppContainer(param.Parameterized):
    """Reactive container that switches between login and authenticated app."""
    
    authenticated = param.Boolean(default=False)
    username = param.String(default="")
    error_message = param.String(default="")
    
    def __init__(self, **params):
        super().__init__(**params)
    
    def _login(self, username: str, password: str):
        """Handle login attempt."""
        if not username or not password:
            self.error_message = "❌ Please enter username and password"
            return
        
        try:
            if auth_manager.authenticate(username, password):
                session_id = auth_manager.create_session(username)
                pn.state.cache["session_id"] = session_id
                pn.state.cache["username"] = username
                self.username = username
                self.authenticated = True  # Explicitly set to boolean True
                self.error_message = ""
                logger.info(f"User {username} authenticated")
            else:
                self.error_message = "❌ Invalid username or password"
        except Exception as e:
            self.error_message = f"❌ Error: {str(e)}"
            logger.error(f"Authentication error: {e}")
    
    def _logout(self):
        """Handle logout."""
        session_id = pn.state.cache.get("session_id")
        if session_id:
            auth_manager.invalidate_session(session_id)
        pn.state.cache.clear()
        self.authenticated = False  # Explicitly set to boolean False
        self.username = ""
        self.error_message = ""
        logger.info("User logged out")
    
    def _create_login_panel(self):
        """Create login UI."""
        username_input = pn.widgets.TextInput(name="Username", placeholder="Enter username")
        password_input = pn.widgets.PasswordInput(name="Password", placeholder="Enter password")
        login_button = pn.widgets.Button(name="Login", button_type="success")
        
        def on_login_click(event):
            self._login(username_input.value, password_input.value)
            password_input.value = ""
        
        login_button.on_click(on_login_click)
        
        def on_password_change(event):
            """Trigger login on Enter key in password field."""
            # Only trigger login if password is not empty
            if password_input.value:
                self._login(username_input.value, password_input.value)
                password_input.value = ""
        
        # Watch password field with onlychanged=True to avoid recursion
        password_input.param.watch(on_password_change, 'value', onlychanged=True)
        
        @pn.depends(self.param.error_message)
        def update_error(error_message):
            return pn.pane.Markdown(error_message)
        
        return pn.Column(
            pn.pane.Markdown("# Map Visualizer Login"),
            pn.pane.Markdown("Enter your credentials to access the map"),
            username_input,
            password_input,
            login_button,
            update_error,
            width=400,
            align="center",
        )
    
    @pn.depends("authenticated")
    def get_content(self):
        """Get the appropriate content based on auth state."""
        if not self.authenticated:
            return self._create_login_panel()
        else:
            # Create app for authenticated user with logout callback
            app = MapVisualizerApp(username=self.username, on_logout=self._logout)
            return app.panel()


def create_login_page():
    """Create a login page."""
    username_input = pn.widgets.TextInput(name="Username", placeholder="Enter username")
    password_input = pn.widgets.PasswordInput(name="Password", placeholder="Enter password")
    login_button = pn.widgets.Button(name="Login", button_type="success")
    error_message = pn.pane.Markdown("")
    
    def on_login_click(event):
        username = username_input.value
        password = password_input.value
        
        if not username or not password:
            error_message.object = "❌ Please enter username and password"
            return
        
        try:
            if auth_manager.authenticate(username, password):
                session_id = auth_manager.create_session(username)
                pn.state.cache["session_id"] = session_id
                pn.state.cache["username"] = username
                pn.state.curdoc.add_next_tick_callback(
                    lambda: pn.io.notebook.push_notebook() if hasattr(pn.io, 'notebook') else None
                )
            else:
                error_message.object = "❌ Invalid username or password"
                password_input.value = ""
        except Exception as e:
            error_message.object = f"❌ Error: {str(e)}"
            logger.error(f"Authentication error: {e}")
    
    login_button.on_click(on_login_click)
    
    return pn.Column(
        pn.pane.Markdown("# Map Visualizer Login"),
        pn.pane.Markdown("Enter your credentials to access the map"),
        username_input,
        password_input,
        login_button,
        error_message,
        width=400,
        align="center",
    )


# Global app instance
_app_container = None

def get_app_container() -> AppContainer:
    """Get or create the global app container."""
    global _app_container
    if _app_container is None:
        # Check if user is already authenticated via cache
        session_id = pn.state.cache.get("session_id")
        username = pn.state.cache.get("username")
        is_authenticated = bool(session_id and auth_manager.is_authenticated(session_id))
        
        _app_container = AppContainer(
            authenticated=is_authenticated,
            username=username or ""
        )
    return _app_container


def create_panel_app():
    """Create the Panel app with reactive authentication."""
    container = get_app_container()
    
    # Create a function that depends on authenticated state
    @pn.depends(container.param.authenticated)
    def _render_app(authenticated):
        return container.get_content()
    
    return pn.Column(_render_app)


@add_application('/panel', app=app, title='Map Visualizer')
def panel_app():
    """Integrate Panel app with FastAPI."""
    try:
        return create_panel_app()
    except Exception as e:
        logger.error(f"Error creating Panel app: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pn.pane.Markdown(f"# Error\n\n```\n{traceback.format_exc()}\n```")


def start():
    """Start the FastAPI server with embedded Panel and tile serving."""
    import uvicorn
    
    host = os.getenv("MAPDISPLAY_HOST", default="localhost")
    host = "0.0.0.0" if Path("/.dockerenv").exists() else host
    port = int(os.getenv("MAPDISPLAY_PORT", default="5006"))
    workers = int(os.getenv("MAPDISPLAY_WORKERS", default="1"))

    logger.info(f"Starting FastAPI server with embedded Panel at {host}:{port}")
    logger.info(f"Panel UI available at http://{host}:{port}/panel")
    logger.info(f"Tile API available at http://{host}:{port}/tile/{{var}}/{{species}}/{{time}}/{{z}}/{{x}}/{{y}}")
    logger.info("Note: Hot reload requires running via CLI: uvicorn md.ui.display:app --reload")
    
    # Note: reload is incompatible with passing app object directly to uvicorn.run()
    # It only works with import strings
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )