FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        python3-gdal \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

COPY compare_catalogs_dashboard.py . 

RUN pip install --no-cache-dir dash plotly obspy numpy pandas xarray pyproj shapely gcsfs "gdal==3.10.3" gunicorn

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "compare_catalogs_dashboard:app"]