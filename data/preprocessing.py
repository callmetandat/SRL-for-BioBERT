import os
import spacy
import pandas as pd
from tqdm import tqdm
from xml.dom import minidom

PATH_DATA = '../data/'

# create class to preprocess data
class Preprocessing:
   def __init__(self, file_name):
      self.file_name = file_name
      self.data_arg = None
      self.predicate = None
      self.roles = []
      self.data_role = None
      self.min_threshold = 0.1
      self.output_data = PATH_DATA + 'interim/' + file_name.split('.')[0] + '.csv'
      self.path_statistic_data = PATH_DATA + 'statistics/' + file_name.split('.')[0] + '.csv'
      
   def read_xml_file(self):
      mydoc = minidom.parse(PATH_DATA + 'raw/' + self.file_name)
      self.predicate = mydoc.getElementsByTagName('predicate')[0].getAttribute('lemma')
      self.roles = dict()
      for arg in mydoc.getElementsByTagName('role'):
         self.roles.update({arg.getAttribute('n'): arg.getAttribute('descr')})
      examples = mydoc.getElementsByTagName('example')
      ids = [i for i in range(len(examples))]
      srcs, texts, args = [], [], []
      for example in examples:
         text = example.getElementsByTagName('text')[0].firstChild.nodeValue
         src = example.getAttribute('src')
         arg_temp = dict()
         for arg in example.getElementsByTagName('arg'):
            arg_temp.update({arg.getAttribute('n'): arg.firstChild.nodeValue})
         texts.append(text)
         srcs.append(src)
         args.append(arg_temp)
      self.data_arg = pd.DataFrame({'id': ids, 'source': srcs, 'text': texts, 'arguments': args})

   def __remove_argument__(self, index_role):
      if index_role < 0 or index_role >= len(self.roles):
         return
      for i in range(len(self.data_arg['arguments'])):
         if list(self.roles.items())[index_role][0] in self.data_arg['arguments'][i]:
            self.data_arg['arguments'][i].pop(list(self.roles.items())[index_role][0])
   
   def statistic_arg(self):
      args = []
      dependencies = []
      count_examples = []
      nlp = spacy.load('en_core_web_sm')
      for i in range(len(self.data_arg)):
         doc = nlp(self.data_arg['text'][i])
         root = [token for token in doc if token.head == token][0]
         for token in doc:
            if token.head.text == root.text and token.dep_ not in dependencies:
               dependencies.append(token.dep_)
               count_examples.append(0)
         for j in range(len(self.data_arg['arguments'][i])):
            if list(self.data_arg['arguments'][i].items())[j][0] not in args:
               args.append(list(self.data_arg['arguments'][i].items())[j][0])
      for i in range(len(self.data_arg)):
         doc = nlp(self.data_arg['text'][i])
         root = [token for token in doc if token.head == token][0]
         for token in doc:
            for j in range(len(self.data_arg['arguments'][i])):
               if token.text in list(self.data_arg['arguments'][i].items())[j][1] and token.head.text == root.text:
                  count_examples[dependencies.index(token.dep_)] += 1
      
      data_role_temp = []
      for i in range(len(args)):
         for j in range(len(dependencies)):
            data_role_temp.append([args[i], dependencies[j], count_examples[j]])
      self.data_role = pd.DataFrame(data_role_temp, columns=['arg', 'dependency', 'count'])
      self.data_role.to_csv(self.path_statistic_data, index=False)
      
   def dependency_parsing(self):
      def print_dependency_parsing(token):
         print(
            f"""
               TOKEN: {token.text}
               =====
               {token.tag_ = }
               {token.head.text = }
               {token.dep_ = }
               {spacy.explain(token.dep_) = }""")
      
      max_len_arg = max([len(arg) for arg in self.data_arg['arguments'].values])
      count_args = [0 for i in range(max_len_arg)]
      nlp = spacy.load('en_core_web_sm')
      lst_index_remove = []
      for i in tqdm(range(len(self.data_arg))):
         doc = nlp(self.data_arg['text'][i])
         root = [token for token in doc if token.head == token][0]
         for token in doc:
            for j in range(len(self.data_arg['arguments'][i])):
               if token.text in list(self.data_arg['arguments'][i].items())[j][1] and token.head.text == root.text:
                  count_args[j] += 1
      for j in range(len(count_args)):
         if count_args[j] < len(self.data_arg) * self.min_threshold:
            lst_index_remove.append(j)
      for index in sorted(lst_index_remove, reverse=True):
         self.__remove_argument__(index)
      self.data_arg.to_csv(self.output_data, index=False)
      self.statistic_arg()

filenames = os.listdir(PATH_DATA + 'raw')
for filename in filenames:
   print(filename)
   preprocessor = Preprocessing(filename)
   preprocessor.read_xml_file()
   preprocessor.dependency_parsing()



