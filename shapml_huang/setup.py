import setuptools
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding="utf8").read()

base_packages = [
# "numpy==1.19.5",
"tornado>=6.1",
#"numba==0.52.0",
"ipykernel",
"Boruta==0.3",
"pandas==1.2.5",
"xgboost==1.3.3",
"hyperopt==0.2.5",
"bokeh==2.4.1",
"patsy>=0.5.1",
"scikit-learn>=0.23.1",
"seaborn>=0.11.0",
"tqdm>=4.47.0",
"matplotlib>=3.2.2",
"missingno>=0.4.2",
"statsmodels>=0.11.1",
"lifelines>=0.26.0",
"jupyterlab==2.1.5",
"ipywidgets>=7.5.1",
"dowhy>=0.6", 
"hvplot==0.6.0",
"swifter>=1.0.9",
"cachetools>=4.1.1",
"category-encoders>=2.2.2",
"shap==0.37.0",
"scikit-misc>=0.1.3",
"streamlit==1.2.0",
"streamlit-ace==0.1.1",
"rsconnect-python==1.5.4",
"smote-variants>=0.4.0" # This installs tensorflow automatically
#"ipykernel==5.3.1",
]

setuptools.setup(
    name="shapml",
    version="0.1.0",
    description="",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Rashed Harun",
    author_email="harunr@gene.com",
    license="Apache License",
    packages=setuptools.find_packages(),
    package_data={'': ['shapml/utils/streamlit/requirements.txt']},
    include_package_data=True,
    python_requires="<3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Apache License",
        "Operating System :: OS Independent",
    ],
    install_requires=base_packages,
    url="https://github.roche.com/harunr/shapml",
    zip_safe=False,
)