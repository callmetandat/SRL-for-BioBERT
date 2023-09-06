
import json
import argparse
import os
from utils.transform_utils import TransformParams
from utils.data_utils import TRANSFORM_FUNCS
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, required=True)
    
    parser.add_argument('--transform_file', type=str, required=True, default="embedding.yml")
    
    args = parser.parse_args()
    transformParams = TransformParams(args.transform_file)
    
    for transformName, transformFn in transformParams.transformFnMap.items():
        transformParameters = transformParams.transformParamsMap[transformName]
    dataDir = transformParams.readDirMap[transformName]

    assert os.path.exists(dataDir), "{} doesnt exist".format(dataDir)
    saveDir = transformParams.saveDirMap[transformName]
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
 
    for file in transformParams.readFileNamesMap[transformName]:
        #calling respective transform function over file
        TRANSFORM_FUNCS[transformFn](dataDir = dataDir, readFile=file,
                                    wrtDir=saveDir)
 
if __name__ == '__main__':
    main()