from dotenv import load_dotenv
load_dotenv()

import os
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, LogisticRegression
from pyspark.ml.feature import (
    StringIndexer, Tokenizer, StopWordsRemover,
    HashingTF, IDF, IndexToString
)

from huggingface_hub import HfApi


nltk.download('stopwords')


class Model:
    def __init__(self, dataset_path):
        
        self.spark = SparkSession.builder.appName("classification_category").getOrCreate()
        
        self.df = self.spark.read.options(header=True, delimiter=",").csv(dataset_path)
        
        self.stopword = stopwords.words('indonesian')

    def pipeline(self):
        
        tokenizer = Tokenizer(inputCol="judul", outputCol="words")
        stopword_remover = StopWordsRemover(
            inputCol="words",
            outputCol="token",
            stopWords=self.stopword
        )
        hashingTF = HashingTF(inputCol="token", outputCol="rawfeatures")
        idf = IDF(inputCol="rawfeatures", outputCol="features")
        model = NaiveBayes(smoothing=1.0, modelType="multinomial")

        return Pipeline(stages=[tokenizer, stopword_remover, hashingTF, idf, model])

    def train_and_save_model(self, output_path):
        
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        
        label_indexer = StringIndexer(inputCol="kategori", outputCol="label")
        df_indexed = label_indexer.fit(self.df).transform(self.df)

        
        stemming_udf = udf(lambda s: stemmer.stem(s) if s else "", StringType())
        df_final = df_indexed.withColumn("stemming", stemming_udf(df_indexed["judul"]))

        
        base_pipeline = self.pipeline()
        model = base_pipeline.fit(df_final)

        
        label_decoder = IndexToString(
            inputCol="prediction",
            outputCol="category",
            labels=label_indexer.fit(self.df).labels
        )

        final_pipeline = Pipeline(stages=base_pipeline.getStages() + [label_decoder])
        final_model = final_pipeline.fit(df_final)

        
        try:
            final_model.write().overwrite().save(output_path)
            print(f"✅ Model berhasil disimpan di: {output_path}")
        except Exception as e:
            print(f"❌ Error saat menyimpan model: {e}")


if __name__ == "__main__":
    
    dataset_path = "dataset_kategori.csv"
    output_path = "/mnt/c/Users/eBdesk/documents/project_rozi/create_dataset_ml/model"

    
    trainer = Model(dataset_path)
    trainer.train_and_save_model(output_path)

    
    api = HfApi(token=os.getenv('TOKEN_HF'))

    api.create_repo(
        repo_id="Rozirizky/categori_models",
        repo_type="model",
        private=False,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=output_path,
        repo_id="Rozirizky/categori_models",
        repo_type="model",
    )

    print("🚀 Model berhasil diunggah ke Hugging Face Hub!")
