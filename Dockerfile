
# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

#EXPOSE 8501
EXPOSE 8080

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip3 install --no-cache-dir -r  requirements.txt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.

# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app

#CMD streamlit run main.py --server.port=${PORT} --browser.serverAddress="0.0.0.0"
#CMD streamlit run main.py --server.port 8080 --server.enableCORS false

#CMD streamlit run app.py --server.port=${PORT} --browser.serverAddress="0.0.0.0"
CMD streamlit run app.py --server.port 8080 --server.enableCORS false