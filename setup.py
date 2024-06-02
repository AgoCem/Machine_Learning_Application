# here i will just build my machine learning project as a package, so also others can use it :)
from setuptools import find_packages, setup
from typing import List

E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    #this function will give me the list of packages requirements

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines() 
        requirements = [req.replace('\n','') for req in requirements]
        if E_DOT in requirements:
            requirements.remove(E_DOT)

    return requirements
        
        


setup(

    name = 'Machine_Learning_Application',
    version = '0.0.1',
    author='Agostino',
    author_email = 'agostinocembalo@gmail.com',
    packages = find_packages(), #this function will find for __init__.py files and consider the folder as a package
    install_requires = get_requirements('requirements.txt')
)