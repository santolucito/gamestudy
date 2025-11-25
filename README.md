# Game Study

## Running Locally

Audio recording requires HTTPS. Generate a certificate and start the server:

```bash
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"

python3 -m https_server
```

Then visit https://localhost:8443/game.html (accept the certificate warning).
