
#!/bin/bash

# Ensure the script stops on error
set -e

# Set Flask environment
export FLASK_APP=app.py
export FLASK_ENV=production

# Run the Flask app using Render's assigned PORT
flask run --host=0.0.0.0 --port=$PORT
