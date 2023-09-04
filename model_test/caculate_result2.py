import os 
from sklearn.metrics import precision_score, recall_score, f1_score


pathData = "./srl_output"
# pathPredict = "./label"
listfile = os.listdir(pathData)
# listlabel = os.listdir(pathPredict)


# for file in listlabel:
# name1 = "test_delete" #get name file in name.xml
# name1 = name1.split('_')[0] #get verb in case a_1.xml
# print(f"file: {file}, name:{name1}")
# if  "test_" not in name1:
#     continue
# print(f'"{name1.split("_")[1]}"',end=",")

for file in listfile:
    if "test_" not in file:
        continue
    name1 = file.split(".")[0][6:]
    trueLabel = []
    predictLabel = []
    with open(f"{pathData}/token_{name1}.txt","r") as dataFile, open(f"{pathData}/label_{name1}.txt","r") as predictFile:
        data = dataFile.readlines()
        predict = predictFile.readlines()  
        
        # for line in data:
        #     # print(f"data line: {line}, len{len(line)}")
        #     if (line[0:2] == "##") or ("[SEP]" in line) or ("[CLS]" in line) or len(line) < 2:
        #         trueLabel.append("O")
        #         continue

            
        #     trueLabel.append(line.split("\t")[1].strip())
            
        # for line in predict:
        #     # print(f"predict line: {line}")
        #     # if line in ["[CLS]","X"] or len(line) < 2:
        #     if line in ['[CLS]','[SEP]', 'X']:
        #         predictLabel.append("O")
        #         # print(line)
        #         continue
        #     else:
        #         predictLabel.append(line.strip())
        for lineIdx, (lineTok, lineLab) in enumerate(zip(data, predict)):
            lineTok = lineTok.strip()

            lineLab = lineLab.strip()
            if lineLab in ['[CLS]','[SEP]', 'X']: # replace non-text tokens with O. These will not be evaluated.
                predictLabel.append('O')
                trueLabel.append('O')
                continue
            if(lineLab == "B-V"):
                predictLabel.append("V")
            else:
                predictLabel.append(lineLab)
            trueLabel.append(lineTok.split()[1])
            
        
        # trueLabel = trueLabel[0:len(predictLabel)]        

        if len(trueLabel) != len(predictLabel):
            print(f"{name1} trueLabel and predictLabel not have the same len, {len(trueLabel)} vs {len(predictLabel)}")
            exit()
        print(f"{name1} trueLabel and predictLabel not have the same len, {len(trueLabel)} vs {len(predictLabel)}")


    result_p = precision_score(trueLabel,predictLabel,average="micro")
    result_r = recall_score(trueLabel,predictLabel,average="micro")
    result_f1 = f1_score(trueLabel,predictLabel,average="micro")

    with open(f"./result/result_{name1}.txt", "w") as result:
        
        result.write(f"*********Predict predicate {name1} result **********\n")
        result.write(f"precision_score: {result_p}\n")
        result.write(f"recall_score: {result_r}\n")
        result.write(f"f1_score: {result_f1}\n")
        print(f"*********Predict predicate {name1} result **********\n")
        print(f"precision_score: {result_p}\n")
        print(f"recall_score: {result_r}\n")
        print(f"f1_score: {result_f1}\n")















