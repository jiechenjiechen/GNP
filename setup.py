from setuptools import setup, find_packages

setup(
    name='GNP',
    version=1.0,
    author='Jie Chen',
    author_email='chenjie@us.ibm.com',
    description='Graph neural preconditioenr',
    packages=find_packages(),
    install_requires=[
        'mat73',
        'tqdm',
        'numpy',
        'scipy',
        'torch',
        'ssgetpy',
    ],
)
