# Introduction

The log-files are bundled into a single zip file. Focus on ``Heqing_device2`` here. We will:

#### Extraction

- extract the content of each log-file
- select only the system-calls
- store it into a csv for now

#### Encoding

- then read these csvs and fit a TF-IDF Vectorizer
- store the result into csvs again