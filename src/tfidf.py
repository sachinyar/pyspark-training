from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, ArrayType, StringType, IntegerType
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
 

def indices_to_terms(vocabulary):
    def indices_to_terms(xs):
        return [vocabulary[int(x)] for x in xs]
    return udf(indices_to_terms, ArrayType(StringType()))


def main():

    myMovieSchema = StructType([
        StructField("Release Year", IntegerType(), False),
        StructField("Title", StringType(), True),
        StructField("Origin", StringType(), True),
        StructField("Director", StringType(), True),
        StructField("Cast", StringType(), True),
        StructField("Genre", StringType(), True),
        StructField("Wiki-page", StringType(), True),
        StructField("Plot", StringType(), True)
    ])

    movies_df = spark.read.format("csv").option("header", "true").option("multiLine", "true").option("escape", '"').schema(myMovieSchema).load("/home/ubuntu/data/wiki_movie_plots_deduped.csv")
    
    movies_df.show()

    tokenizer = RegexTokenizer(inputCol="Plot", outputCol="all_words", minTokenLength=2, pattern="[^a-zA-Z]")

    remover = StopWordsRemover(inputCol="all_words", outputCol="words", stopWords=StopWordsRemover.loadDefaultStopWords('english'))

    cv = CountVectorizer(inputCol='words', outputCol='rawFeatures', minDF=2.0)

    idf = IDF(inputCol='rawFeatures', outputCol='features')
 
    lda = LDA(k=25, seed=123, optimizer='em', featuresCol='features')

    stages = [tokenizer, remover, cv, idf]

    pipelineModel = Pipeline(stages=stages)

    tfidfmodel = pipelineModel.fit(movies_df)

    vocab = tfidfmodel.stages[2].vocabulary

    print("Here is my vocabulary {}".format(vocab[:10]))

    rescaledData = tfidfmodel.transform(movies_df)

    rescaledData.show()

    rescaledData = rescaledData.drop("words")

    ldamodel = lda.fit(rescaledData)

    ldaTopicIndices = ldamodel.describeTopics()

    ldaTopicIndices.show(20, False)


    ldatopicswords = ldaTopicIndices.withColumn("topics_words", indices_to_terms(vocab)("termIndices"))

    ldatopicswords.select("topics_words").show(25, False)
 

if __name__ == '__main__':

    sc = SparkContext()

    spark = SparkSession.builder.appName('KiMi_TFIDF').getOrCreate()

    main()

    sc.stop()
