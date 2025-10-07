## Run locally
- Download Python3.10
- Create a virtual environment ```py -3.10 -m venv venv```, ```.\venv\Scripts\activate```
- Install dependencies ```pip install -r requirements.txt```
- Build package ```python setup.py build_ext --inplace```, ```python setup.py build_ui```
- Adjust configuration in ```gridsearch_local.py```
- Run ```py -3.10 gridsearch.local.py```, adjust code as needed for postprocessing

## Run Dockerfile
