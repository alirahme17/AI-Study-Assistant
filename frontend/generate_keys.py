"""Print a bcrypt hash for a password, to paste into config.yaml by hand.

Usage:  python generate_keys.py
The password is typed interactively and never stored in this file.
"""
import getpass

import streamlit_authenticator as stauth

password = getpass.getpass("Password to hash: ")
print(stauth.Hasher().hash(password))
