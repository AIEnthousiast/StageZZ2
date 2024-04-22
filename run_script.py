from misc import * 
import sys
import os


def run(filename,time,save_folder):
    splits = filename.split("\\")
    
    log_file_name = f"log_{splits[-1]}"
    
    
    _,edges,_,blocks = read_instance(filename)
    instance = GraphInstance(edges=edges,blockages=blocks)

    model = SNOPCompactModel(instance=instance)
    model.solve(timeLimit=time)
    
    with open(os.path.join(save_folder,log_file_name),"w") as f:
        f.write(f"Instance_name:{splits[-1]}\n")
        f.write(f"Time:{model.model.Runtime}\n")
        f.write(f"Objective:{model.model.ObjBound}\n")
        f.write
   
def run_folder(folder_name,time):
    print(folder_name)
    for file in os.listdir(folder_name):
        file_path = os.path.join(folder_name,file)
        if os.path.isfile(file_path):
            run(file_path,time,folder_name)
        elif os.path.isdir(file_path):
            run_folder(file_path,time)
            
           

if __name__ == "__main__":
    name = sys.argv[1]
    time = float(sys.argv[2])
    print(name)
    if os.path.isfile(name):
        run(name,time,os.getcwd())
    elif os.path.isdir(name):
        print("ok")
        run_folder(name,time)
        
    
    