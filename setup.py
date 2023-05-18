from setuptools import setup


setup(
    name="ehr_diagnosis_env",
    version="0.0.1",
    install_requires=["gymnasium==0.26.0", "torch==2.0.1+cu118", "transformers==4.29.1", "accelerate==0.19.0"],
)
