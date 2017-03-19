#!/bin/bash

# Compile the main.tex file and set garbage output to outputGarbage folder
pdflatex -aux-directory=outputGarbage/ -output-directory=outputGarbage/ main.tex

