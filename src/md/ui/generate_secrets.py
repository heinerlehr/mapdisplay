import csv
import os

# Argument parsing
from argparse import ArgumentParser, Namespace
from pathlib import Path

from cryptography.fernet import Fernet


def load_or_create_key():
    key_fn = Path(os.getenv("SECRETS_FILE", default="secret.key"))
    """Loads existing key or creates a new one."""
    if key_fn.exists():
        with open(key_fn, "rb") as key_file:
            return key_file.read()
    else:
        with open(key_fn, "wb") as key_file:
            key = Fernet.generate_key()
            key_file.write(key)
            return key

def retrieve_credentials(decrypt:bool=True)->dict:
    credentials = {}
    filename = Path(os.getenv("CREDENTIALS_FILE", default="credentials.csv"))
    if not filename.exists():
        return credentials
    with open(filename, mode="r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            username, password = row
            if decrypt:
                password = decrypt_password(password)
            credentials[username] = password
    return credentials

# 🔹 Encrypt Password
def encrypt_password(password):
    key = load_or_create_key()
    cipher = Fernet(key)
    return cipher.encrypt(password.encode()).decode()

def decrypt_password(encrypted_password):
    key = load_or_create_key()
    cipher = Fernet(key)
    return cipher.decrypt(encrypted_password.encode()).decode()

# 🔹 Save Encrypted Credentials
def save_credentials(username, password):
    credentials = retrieve_credentials(decrypt=False)
    encrypted_password = encrypt_password(password)
    credentials[username] = encrypted_password
    filename = Path(os.getenv("CREDENTIALS_FILE", default="credentials.csv"))
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        for user, encrypted_password in credentials.items():
            writer.writerow([user, encrypted_password])

def get_args() -> Namespace:
    """Reads command line arguments and returns a Namespace object with them

    Returns:
        Namespace: Namespace object with the command line arguments
    """
    parser = ArgumentParser(
        prog='generate-secrets.py',
        description='Entry point for the Health Monitor that allows training, inference, retraining, and EDA'
    )
    parser.add_argument('--username', type=str, help='Username for the credentials')
    parser.add_argument('--password', type=str, help='Password for the credentials')
    parser.add_argument('--quiet', action='store_true', help='If to produce output')
    return parser.parse_args()


# 🔹 Example Usage
if __name__ == "__main__":

    args = get_args()
    if not args.username:
        username = input("Enter username: ")
    else:
        username = args.username

    if not args.password:
        password = input("Enter password: ")
    else:
        password = args.password

    # Save the credentials
    save_credentials(username, password)

    # Retrieve and display all stored credentials
    if not args.quiet:
        print("\n🔐 Stored Credentials for users")
        for user, decrypted_pw in retrieve_credentials().items():
            print(f"👤 {user}")
