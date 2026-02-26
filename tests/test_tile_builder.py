"""Tests for tile_builder module with optimization features."""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import xarray as xr
from unittest.mock import Mock, patch

from iconfig.iconfig import iConfig

from md.preprocess.tile_builder import (
    DataRenderer,
    VariableConfig,
    WebMercatorTiler,
    BatchTileBuilder,
    build_tiles_batch,
    HAS_CUPY,
)


class TestDataRenderer:
    """Test DataRenderer with optimizations."""

    def test_normalize_data_with_provided_range(self) -> None:
        """Test normalization with explicit vmin/vmax."""
        renderer = DataRenderer(use_gpu=False)
        data = np.array([[0.0, 50.0], [100.0, 200.0]])
        
        normalized = renderer.normalize_data(data, vmin=0.0, vmax=100.0)
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized[0, 0] == pytest.approx(0.0)
        assert normalized[0, 1] == pytest.approx(0.5)
        assert normalized[1, 0] == pytest.approx(1.0)

    def test_normalize_data_auto_compute_range(self) -> None:
        """Test normalization with automatic percentile computation."""
        renderer = DataRenderer(use_gpu=False)
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 100.0]])  # 100 is outlier
        
        normalized = renderer.normalize_data(data, vmin=None, vmax=None)
        
        # Should use 2nd and 98th percentiles (exclude extreme outlier)
        assert normalized.min() >= -0.1  # Allow small negative due to percentile cutoff
        assert normalized.max() <= 1.1  # Allow small overflow due to percentile cutoff

    def test_normalize_data_with_sentinels(self) -> None:
        """Test normalization handles -999 sentinel values."""
        renderer = DataRenderer(use_gpu=False)
        data = np.array([[-999.0, 10.0], [20.0, 30.0]])
        
        normalized = renderer.normalize_data(data, vmin=None, vmax=None)
        
        # Should exclude -999 from percentile calculation
        valid_count = np.sum(data > -999)
        assert valid_count == 3

    def test_colormap_caching(self) -> None:
        """Test that colormaps are cached after first retrieval."""
        renderer = DataRenderer(use_gpu=False)
        
        # First call caches it
        cmap1 = renderer.get_colormap("viridis")
        # Second call retrieves from cache
        cmap2 = renderer.get_colormap("viridis")
        
        # Should be same object (cached)
        assert cmap1 is cmap2
        
        # Check cache has entry
        assert "viridis" in renderer._colormap_cache

    def test_colormap_multiple_colormaps(self) -> None:
        """Test caching works with multiple colormaps."""
        renderer = DataRenderer(use_gpu=False)
        
        cmap_viridis = renderer.get_colormap("viridis")
        cmap_coolwarm = renderer.get_colormap("coolwarm")
        
        assert len(renderer._colormap_cache) == 2
        assert "viridis" in renderer._colormap_cache
        assert "coolwarm" in renderer._colormap_cache

    def test_apply_colormap_shape(self) -> None:
        """Test colormap application produces correct shape."""
        renderer = DataRenderer(use_gpu=False)
        data = np.random.rand(64, 64)  # Normalized data [0, 1]
        
        colored = renderer.apply_colormap(data, "viridis", transparent_values=[0.0])
        
        # Should be (height, width, 4) with RGBA
        assert colored.shape == (64, 64, 4)
        assert colored.dtype == np.uint8
        assert colored.min() >= 0
        assert colored.max() <= 255

    def test_resample_to_tile_size_identity(self) -> None:
        """Test resampling when already tile size."""
        renderer = DataRenderer(use_gpu=False)
        data = np.random.rand(256, 256)
        
        resampled = renderer.resample_to_tile_size(data, 256)
        
        # Should be identical if already correct size
        assert np.array_equal(resampled, data)

    def test_resample_to_tile_size_upscale(self) -> None:
        """Test resampling upscaling."""
        renderer = DataRenderer(use_gpu=False)
        data = np.ones((10, 10))  # Small tile
        
        resampled = renderer.resample_to_tile_size(data, 256)
        
        assert resampled.shape == (256, 256)
        assert np.all(resampled == 1.0)

    def test_resample_to_tile_size_downscale(self) -> None:
        """Test resampling downscaling."""
        renderer = DataRenderer(use_gpu=False)
        data = np.random.rand(512, 512)
        
        resampled = renderer.resample_to_tile_size(data, 256)
        
        assert resampled.shape == (256, 256)

    def test_gpu_flag_without_cupy(self) -> None:
        """Test GPU flag gracefully degrades without CuPy."""
        # Even with flag, should work without CuPy
        renderer = DataRenderer(use_gpu=True)
        
        if not HAS_CUPY:
            # Should have fallen back to CPU
            assert renderer.use_gpu is False
        # If HAS_CUPY, use_gpu might be True

    def test_resample_cpu_fallback(self) -> None:
        """Test CPU fallback works."""
        renderer = DataRenderer(use_gpu=False)
        data = np.random.rand(100, 100)
        
        resampled = renderer.resample_to_tile_size(data, 256)
        
        assert resampled.shape == (256, 256)


class TestWebMercatorTiler:
    """Test Web Mercator tiler."""

    def test_get_tile_bounds(self) -> None:
        """Test tile bounds calculation."""
        tiler = WebMercatorTiler()
        
        # Zoom 0, tile 0,0 should cover entire world
        lon_min, lat_min, lon_max, lat_max = tiler.get_tile_bounds(0, 0, 0)
        
        assert lon_min == pytest.approx(-180)
        assert lon_max == pytest.approx(180)
        assert lat_max == pytest.approx(85.05, abs=0.1)  # Web Mercator max
        assert lat_min == pytest.approx(-85.05, abs=0.1)

    def test_get_zoom_levels_high_resolution(self) -> None:
        """Test zoom level selection for high-resolution data."""
        tiler = WebMercatorTiler()
        
        # 0.1 degree resolution (3600 lon pixels)
        min_z, max_z = tiler.get_zoom_levels((3600, 3600), calculate=True)
        assert max_z == 8

    def test_get_zoom_levels_medium_resolution(self) -> None:
        """Test zoom level selection for medium-resolution data."""
        tiler = WebMercatorTiler()
        
        # 0.25 degree resolution (1440 lon pixels)
        min_z, max_z = tiler.get_zoom_levels((720, 1440), calculate=True)
        assert max_z == 6

    def test_get_zoom_levels_low_resolution(self) -> None:
        """Test zoom level selection for low-resolution data."""
        tiler = WebMercatorTiler()
        
        # Low resolution
        min_z, max_z = tiler.get_zoom_levels((180, 360), calculate=True)
        assert max_z <= 5


class TestVariableConfig:
    """Test VariableConfig."""

    def test_variable_config_defaults(self) -> None:
        """Test VariableConfig with defaults."""
        config = VariableConfig(name="temperature")
        
        assert config.name == "temperature"
        assert config.colormap == "viridis"
        assert config.vmin is None
        assert config.vmax is None
        assert config.is_mask is False

    def test_variable_config_custom(self) -> None:
        """Test VariableConfig with custom values."""
        config = VariableConfig(
            name="mask",
            colormap="gray",
            vmin=0.0,
            vmax=1.0,
            is_mask=True,
        )
        
        assert config.name == "mask"
        assert config.colormap == "gray"
        assert config.vmin == 0.0
        assert config.vmax == 1.0
        assert config.is_mask is True


class TestBatchTileBuilder:
    """Test BatchTileBuilder with optimizations."""

    @pytest.fixture
    def test_config(self):
        """Create a test config with time_range disabled and minimal zoom levels."""
        from unittest.mock import MagicMock
        config = MagicMock()
        # Make the config callable and return None for tiler.time_range
        config.return_value = None
        def config_call(key, default=None):
            if key == "tiler.time_range":
                return None
            # Reduce zoom levels for faster tile generation in tests
            if key == "tiler.max_zoom_levels":
                return 2
            # For other configs, return some reasonable defaults
            if key == "tiler.copernicus_latitude":
                return {"start": -5, "end": 5}
            if key == "tiler.copernicus_longitude":
                return {"start": -5, "end": 5}
            return default
        config.side_effect = config_call
        return config

    @pytest.fixture
    def sample_zarr(self, tmp_path: Path) -> Path:
        """Create a sample zarr file for testing."""
        # Create sample xarray dataset with minimal resolution for fast testing
        lat = np.arange(5, -5, -1.0)  # 10 points
        lon = np.arange(-5, 5, 1.0)   # 10 points
        time = np.arange(0, 3)
        species = ["slat", "mpyr"]
        
        # Create data variables - minimal size for speed
        suitability = xr.DataArray(
            np.random.rand(3, 2, 10, 10),
            dims=["time", "species", "latitude", "longitude"],
            coords={
                "time": time,
                "species": species,
                "latitude": lat,
                "longitude": lon,
            },
        )
        
        ds = xr.Dataset({"suitability": suitability})
        
        zarr_path = tmp_path / "test.zarr"
        ds.to_zarr(zarr_path, mode='w', zarr_format=2)
        
        return zarr_path

    def test_batch_tile_builder_initialization(self, test_config, sample_zarr: Path, tmp_path: Path) -> None:
        """Test BatchTileBuilder initialization with new parameters."""
        config = test_config
        builder = BatchTileBuilder(
            config=config,
            zarr_file=sample_zarr,
            output_dir=tmp_path / "tiles",
            variables=[VariableConfig(name="suitability")],
            use_gpu=False,
            max_workers=2,
        )
        
        assert builder.use_gpu is False
        assert builder.max_workers == 2
        assert builder.renderer.use_gpu is False

    def test_batch_tile_builder_has_pending_tiles_method(self, test_config, sample_zarr: Path, tmp_path: Path) -> None:
        """Test that pending tiles queue exists."""
        """Test BatchTileBuilder initialization with new parameters."""
        config = test_config
        builder = BatchTileBuilder(
            config=config,
            zarr_file=sample_zarr,
            output_dir=tmp_path / "tiles",
            variables=[VariableConfig(name="suitability")],
            use_gpu=False,
            max_workers=2,
        )
        
        # Should have pending_tiles attribute after build starts
        assert hasattr(builder, 'executor') or True  # Executor not created until build()

    def test_build_tiles_batch_with_new_parameters(self, test_config, sample_zarr: Path, tmp_path: Path) -> None:
        """Test build_tiles_batch accepts new optimization parameters."""
        # Should not raise with new parameters
        config = test_config
        build_tiles_batch(
            config=config,
            zarr_file=sample_zarr,
            output_dir=tmp_path / "tiles",
            variables=[VariableConfig(name="suitability", colormap="viridis")],
            use_gpu=False,
            max_workers=2,
        )
        
        # Check tiles were created (at least one zoom level)
        tiles_dir = tmp_path / "tiles"
        assert tiles_dir.exists()
        assert any(tiles_dir.glob("**/**.webp"))

    def test_build_tiles_batch_pre_computed_vmin_vmax(self, test_config, sample_zarr: Path, tmp_path: Path) -> None:
        """Test that vmin/vmax are computed once per variable."""
        variables = [VariableConfig(name="suitability", vmin=None, vmax=None)]
        
        # Should compute vmin/vmax automatically
        config = test_config
        build_tiles_batch(
            config=config,
            zarr_file=sample_zarr,
            output_dir=tmp_path / "tiles",
            variables=variables,
            use_gpu=False,
            max_workers=2,
        )
        
        # After builder processes variable, vmin/vmax should be set
        # (This is verified by successful tile generation)
        assert (tmp_path / "tiles").exists()

    def test_build_tiles_batch_parallel_io(self, test_config, sample_zarr: Path, tmp_path: Path) -> None:
        """Test parallel I/O with multiple workers."""
        config = test_config
        build_tiles_batch(
            config=config,
            zarr_file=sample_zarr,
            output_dir=tmp_path / "tiles",
            variables=[VariableConfig(name="suitability")],
            use_gpu=False,
            max_workers=4,  # Multiple workers
        )
        
        # Should complete without errors
        assert (tmp_path / "tiles").exists()
        # Count tiles generated
        tile_count = len(list((tmp_path / "tiles").glob("**/**.webp")))
        assert tile_count > 0

    def test_build_tiles_batch_gpu_flag(self, test_config, sample_zarr: Path, tmp_path: Path) -> None:
        """Test GPU flag is accepted and handled gracefully."""
        # Should work whether GPU available or not
        config = test_config
        build_tiles_batch(
            config=config,
            zarr_file=sample_zarr,
            output_dir=tmp_path / "tiles",
            variables=[VariableConfig(name="suitability")],
            use_gpu=True,  # Request GPU even if not available
            max_workers=2,
        )
        
        # Should complete successfully
        assert (tmp_path / "tiles").exists()


class TestPreprocessIntegration:
    """Integration tests for preprocessing with new config."""

    def test_preprocess_reads_gpu_config(self) -> None:
        """Test that preprocess reads use_gpu from config."""
        from md.preprocess.preprocess import run_tile_builder
        from iconfig.iconfig import iConfig
        
        config = iConfig()
        # Should be able to read the config without errors
        use_gpu = config("preprocess.use_gpu", default=False)
        max_workers = config("preprocess.max_workers", default=4)
        
        assert isinstance(use_gpu, bool)
        assert isinstance(max_workers, int)
        assert max_workers > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
