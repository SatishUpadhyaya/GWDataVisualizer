FROM gcr.io/google-appengine/python 
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT [ "python", "main.py" ]