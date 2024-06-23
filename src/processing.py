"""
Source: https://csyhuang.github.io/2020/08/01/custom-transformer/
"""
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType
import pyspark.sql.functions as F
from itertools import combinations

class LogTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
  input_col = Param(Params._dummy(), "input_col", "input column name.", typeConverter=TypeConverters.toString)
  output_col = Param(Params._dummy(), "output_col", "output column name.", typeConverter=TypeConverters.toString)
  
  @keyword_only
  def __init__(self, input_col: str = "input", output_col: str = "output"):
    super(LogTransformer, self).__init__()
    self._setDefault(input_col=None, output_col=None)
    kwargs = self._input_kwargs
    self.set_params(**kwargs)
    
  @keyword_only
  def set_params(self, input_col: str = "input", output_col: str = "output"):
    kwargs = self._input_kwargs
    self._set(**kwargs)
    
  def get_input_col(self):
    return self.getOrDefault(self.input_col)
  
  def get_output_col(self):
    return self.getOrDefault(self.output_col)
  
  def _transform(self, df: DataFrame):
    input_col = self.get_input_col()
    output_col = self.get_output_col()
    return df.withColumn(output_col, F.log(input_col))

class MultiplicationTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
  input_cols = Param(Params._dummy(), "input_cols", "input column name.", typeConverter = TypeConverters.toList)
  output_cols = Param(Params._dummy(), "output_cols", "output column name.", typeConverter = TypeConverters.toList)
  
  @keyword_only
  def __init__(self, input_cols: list = [], output_cols: list = []):
    super(MultiplicationTransformer, self).__init__()
    # Use _set to set the parameter values instead of _setDefault
    kwargs = self._input_kwargs
    self.set_params(**kwargs)
    
  @keyword_only
  def set_params(self, input_cols: list = [], output_cols: list = []):
    # Set parameters using _set
    self._set(input_cols=input_cols, output_cols=output_cols) 
    
  def get_input_cols(self):
    return self.getOrDefault(self.input_cols)
  
  def get_output_cols(self):
    return self.getOrDefault(self.output_cols)
  
  def _transform(self, df: DataFrame):
    input_cols = self.get_input_cols()  # Access input_cols as parameter
    output_cols = self.get_output_cols() # Access output_cols as parameter
    for combination in list(combinations(input_cols, 2)):
      df = df.withColumn(f'{combination[0]}*{combination[1]}', F.col(combination[0]) * F.col(combination[1]))
      output_cols.append(f'{combination[0]}*{combination[1]}')
    return df