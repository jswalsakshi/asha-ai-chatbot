import ssl
import certifi
import requests

# Create an SSL context with the correct CA bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Use the context in requests
response = requests.get('https://huggingface.co', headers={'User-Agent': 'my-app'}, verify=ssl_context)
