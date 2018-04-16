# AI_playground
A set of demos for teaching and introducing students to advanced AI-models with few prerequisites.  
  
------
### Repository Structure
The main structure of the repository is:
```
AI_playground/
    notebooks/          <- Interactive notebooks for teaching and showing.
        figures/        
        
    src/                <- Content of main code and tools.
        real_time/      <- Generic tools for running real-time systems.
        image/  
        text/
        audio/
```

##### Running Code
All scripts are run from the top directory `AI_playground`. Examples of how to run a script on Windows and Linux
are shown below, using the snapshot.py-script which opens the webcam for storing photos.  

To run a script in the command line on **Windows** type:
``` 
SET PYTHONPATH=.
python src\image\video\snapshot.py
``` 

To run a script in a terminal on **Linux** type:
``` 
PYTHONPATH=. python src/image/video/snapshot.py
``` 

  
------  
### Comitting Notebooks

**Do not** commit with outputs when committing notebooks.  
In Jupyter, select `Cell -> All Outputs -> Clear` before committing.

 
------
### Links

**Google Drive Link** *(EDIT RIGHTS - don't pass around)*  
https://drive.google.com/drive/folders/1jaUh1Y-TQ95v63luJFC6aQYPni72vg43?usp=sharing

**GitHub Links**
- Project Board: https://github.com/DTUComputeCognitiveSystems/AI_playground/projects/1
- GitHub Pages?: -
