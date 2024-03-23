#!/usr/bin/bash
cd docs/
make all

cd _build/
python3 -m http.server