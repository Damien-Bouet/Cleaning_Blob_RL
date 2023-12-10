import os
from typing import Union, Optional, Dict, List, Any
import pandas as pd
import tensorflow as tf
import numpy as np
import glob
import argparse


class FakeTensorboard:
    """
    This class allow to act like a tensorboard, to store the training data,
    even if the regular tensorboard system is not working when the port connection
    with Unity is used.
    """
    def __init__(self, log_dir: str) -> None:
        """
        Initialize a fake tensorboard.

        Parameters:
        - log_dir (str): The directory to save the fake tensorboard logs.

        Returns:
        None
        """
        self.data : Dict[str, List[List[Union[float,int], int]]] = {}
        self.log_dir = log_dir
        if(log_dir[-1] != "/"):
            self.log_dir += "/"

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    
    def add_scalar(self, name: str, value: Union[float, int], step: int) -> None:
        """
        Add a scalar value to the fake tensorboard.

        Parameters:
        - name (str): The name of the scalar value.
        - value (Union[float, int]): The scalar value.
        - step (int): The step at which the value is recorded.

        Returns:
        None
        """
        if(name in self.data):
            self.data[name].append([value,step])
        else:
            self.data[name] = [[value,step]]


    def to_tensorboard(self) -> None:
        """
        Convert the fake tensorboard data to a tensorboard format.

        Returns:
        None
        """
        print("creation of the tensorboard at "+str(self.log_dir)+"...")
        writer = tf.summary.create_file_writer(self.log_dir)
        for name, values in self.data.items():
            for value,step in values:
                if(value !=""):
                    with writer.as_default():
                        tf.summary.scalar(name,value,step=step)
                        writer.flush()
    

    def save_to_csv(self, path: Optional[str] = None) -> None:
        """
        Save the fake tensorboard data to a CSV file.

        Parameters:
        - path (Optional[str]): The directory to save the CSV file.

        Returns:
        None
        """
        if(path is not None and path[-1] != "/"):
            path += "/"

        res : Dict[str, List[Union[float, int]]] = {}
        for name, values in self.data.items():
            res[str(name)+"_value"]=[]
            res[str(name)+"_step"]=[]
            for value,step in values:
                res[str(name)+"_value"].append(value)
                res[str(name)+"_step"].append(step)
        df = pd.DataFrame(data={})
        for name, value in res.items():
            df = pd.concat([df,pd.DataFrame(data={name:value})],axis=1)
        if(path is None):
            df.to_csv(str(self.log_dir)+"out.csv",index=False)
        else:
            df.to_csv(str(path)+"out.csv",index=False)


    def load_csv(self, path: Optional[str] = None) -> None:
        """
        Load fake tensorboard data from a CSV file.

        Parameters:
        - path (Optional[str]): The directory from which to load the CSV file.

        Returns:
        None
        """
        if(path is None):
            df = pd.read_csv(str(self.log_dir)+"out.csv",index_col=False)
        else:
            df = pd.read_csv(str(path)+"out.csv",index_col=False)
        
        temp : Dict[str, List[List[Any]]] = {}
        for name in df.keys():
            for i in range(len(df[name])):
                if(name.split("_")[1] == "value"):
                    true_name = name.split("_")[0]
                    if(true_name in temp):
                        temp[true_name].append([df[name][i],df[str(true_name)+"_step"][i]])
                    else:
                        temp[true_name] = [ [df[name][i],df[str(true_name)+"_step"][i]] ] 
        
        for name, values in temp.items():
            for value,step in values:
                if(not np.isnan(value)):
                    self.add_scalar(name,value,step)


    def delete_tensorboard(self):
        """
        Delete TensorBoard files with a '.v2' extension from the specified log directory.
        This function removes TensorBoard files that may have been generated during training.

        Returns:
        None
        """
        for v2path in glob.iglob(os.path.join(self.log_dir, '*.v2')):
            os.remove(v2path)


    def reset(self) -> None:
        """
        Reset the fake tensorboard data.

        Returns:
        None
        """
        self.data = {}



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', type=str, default="none", help='Path to out.csv')
    args = parser.parse_args()

    if(args.path != "none"):
        tb = FakeTensorboard(args.path)
        tb.load_csv()
        tb.delete_tensorboard()
        tb.to_tensorboard()
        print("------------------------------------------------------------------------")
        print("Tensorboard created, now run : tensorboard --logdir "+str(args.path))
        print("-----------------------------------------------------------------------")