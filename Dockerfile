FROM python:3.7.9
COPY . .
RUN pip install -e .
CMD python app/main.py