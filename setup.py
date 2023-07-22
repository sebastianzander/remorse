from setuptools import setup, find_packages

setup(
    name = 'remorse',
    version = '0.2.0',
    description = ('Remorse is a command line utility program and library that allows to convert between plain text '
                   'strings, Morse code strings and Morse code sound'),
    author = 'Sebastian Zander',
    author_email = 'remorse@sebastianzander.de',
    url = 'https://github.com/sebastianzander/remorse',
    license = 'MIT license',
    keywords = 'morse morse-code morse-text morse-sound morse-translator morse-converter morse-conversion',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Education',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Telecommunications Industry',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: Android',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Other OS',
        'Operating System :: Unix',
        'Operating System :: iOS',
        'Programming Language :: Python',
        'Topic :: Communications',
        'Topic :: Education',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Utilities'
    ],
    packages = find_packages(exclude = ['*.tests']),
    install_requires = [
        'matplotlib',
        'numpy',
        'pyaudio',
        'scikit-learn',
        'scipy',
        'soundfile',
    ],
    entry_points = {
        'console_scripts': [
            'remorse = remorse.main:main',
        ],
    },
)
