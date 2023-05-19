from setuptools import setup


setup(
    name="ehr_diagnosis_env",
    version="0.0.1",
    install_requires=["gymnasium==0.28.1", "transformers==4.29.1", "accelerate==0.19.0", "pandas==2.0.1",
                      "sentencepiece==0.1.99", "sentence-transformers==2.2.2"],
)
