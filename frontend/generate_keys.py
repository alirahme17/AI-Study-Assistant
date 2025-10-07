import streamlit_authenticator as stauth

# Dictionary of usernames and their plain-text passwords
passwords_to_hash = {
    "user_a": "pass123",
    "user_b": "pass456"
}

# Create the hasher object
hasher = stauth.Hasher()

# Hash each password individually
hashed_passwords = [hasher.hash(pw) for pw in passwords_to_hash.values()]

# Print YAML-ready credentials
print("Copy the following into your config.yaml file:")
print("credentials:")
print("  usernames:")
for i, username in enumerate(passwords_to_hash.keys()):
    print(f"    {username}:")
    print(f"      email: {username}@example.com")
    print(f"      name: User {username.split('_')[1].upper()}")
    print(f"      password: '{hashed_passwords[i]}'")
