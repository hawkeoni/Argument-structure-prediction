FROM python:3.7.9
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .
CMD python argmining/app/main.py