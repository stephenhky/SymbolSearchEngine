
from setuptools import setup


def install_requirements():
    return [package_string.strip() for package_string in open('requirements.txt', 'r')]

def package_description():
    return open('README.md', 'r').read()


setup(name='symutils',
      version='0.0.1a1',
      description="Stock Symbol Search",
      long_description=package_description(),
      long_description_content_type='text/markdown',
      classifiers=[
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Text Processing :: Linguistic",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3.11",
          "Natural Language :: English",
          "License :: OSI Approved :: MIT License",
          "Intended Audience :: Developers",
          "Intended Audience :: Information Technology",
          "Intended Audience :: Financial and Insurance Industry"
      ],
      keywords="symbol search",
      url="https://github.com/stephenhky/SymbolSearchEngine",
      author="Kwan Yuet Stephen Ho",
      author_email="stephenhky@yahoo.com.hk",
      license='MIT',
      packages=['symutils',
                'symutils.ml'],
      package_dir={'symutils': 'symutils'},
      python_requires='>=3.8',
      install_requires=install_requirements(),
      scripts=['script/train_symbolextractor',
               'script/mnb_symbol_nextractor'],
      test_suite="test",
      zip_safe=False)
