/* *****************************************************     */
/* Jincheng Bai, Ryan Murphy, Zizhuang Wu                    */
/* Fall 2015                                                 */
/*  Implement random forest to classify hospital readmission */ 
/* ***************************************************** */

import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini
import scala.collection.breakOut
import scala.collection.generic.CanBuildFrom
import java.io._
import java.util.Calendar


/* -------------------------------------------- */
/* Preamble */
/* -------------------------------------------- */
//// Define parameters here
val MAX_DEPTH = 30
val N_BOOTS : Int = 1
val N_FEATURES : Int = 7
// Define Column Name integer constants
val ROW_INDX : Int = 45 - 1

val INFILE : String = "C:\\a-My_files_and_folders\\1_Purdue\\_Courses\\D_2015_Fall\\Big_Data\\Data\\diabetic_data_cleaned.csv"
//val INFILE : String = "/user/murph213/Project/diabetic_data_cleaned.csv"
val OUTDIR : String = "/home/murph213/Project/OUTPUT/"

// Start Timer
val startTime = Calendar.getInstance.getTime
val start : Long = startTime.getTime

/* -------------------------------------------- */
/* Read and pre-process */
/* -------------------------------------------- */
val data1 = sc.textFile(INFILE) // read in csv

// delete the header
val header = data1.first()
val data2 = data1.filter(x => x !=header)

// Parse data: turn strings into double arrays
val parsedData = data2.map{line =>
    // Create array of doubles
    line.split(',').map(_.toDouble)
}
parsedData.persist()

/* -------------------------------------------- */
/* Ship out bootstrapped samples over the cluster */
/* -------------------------------------------- */

// Create a list of 1, 2, ..., N_BOOTS
val bootIndx =(1 to N_BOOTS).toList
val nrow = parsedData.count.toInt //get size of dataset
val ncol = parsedData.first.length
val featureIndx = (0 to ncol - 3).toList // exclude label and row count

val trains_and_tests = bootIndx.map( x => { 

    ///////// Take a bootstrap sample
    //We can only guarantee getting the desired number
    // of rows if we use "takeSample", which returns array
    // so then we need to parallelize it again
    val tmpArray =  parsedData.takeSample(withReplacement = true, num = nrow)
    val bstrap = sc.parallelize(tmpArray)

    ////////// Get out of bag sample
    // Get row numbers that we took from 
    val bootstrapRows = bstrap.map(line => line(ROW_INDX)).distinct.collect
    
    // Get out of bag sample
    val testds = parsedData.filter( line => !bootstrapRows.contains(line(ROW_INDX)))
    
    ///////// Select subset of features
    // Randomly take numbers from column indicies 0, 1, ..., p-3
    val randCollection = scala.util.Random.shuffle(featureIndx).take(N_FEATURES)

    val trainds = bstrap.map( line => {
        // Create new (smaller) feature vector 
        val someFeatures : Array[Double] = Array()
        val someFeaturesBuff = someFeatures.toBuffer
        for (i <- randCollection){  someFeaturesBuff += line(i) }
        
        // Label the line, for use in decision tree package
        LabeledPoint( line.last, Vectors.dense(someFeaturesBuff.toArray))
    })
    
    // Return
    (trainds,testds)
    
})

val trains = trains_and_tests.map(  ds => ds._1  )
val tests  = trains_and_tests.map(  ds => ds._2  )

trains.foreach(x => x.persist(StorageLevel.MEMORY_AND_DISK))
tests.foreach(x => x.persist(StorageLevel.MEMORY_AND_DISK))

parsedData.unpersist()

/* -------------------------------------------- */
/* Make decision trees for each bootstrapped sample */
/* -------------------------------------------- */

val treeModels = trains.map( treeIn => {
    DecisionTree.train(treeIn, Classification, Gini, MAX_DEPTH)
})

/* -------------------------------------------------- */
/* Run trees against the bootstrap they came from */
/* -------------------------------------------------- */

// Format training datasets with decision tree LabeledPoints
val testsLabeled = tests.map( ds => {
    ds.map( parts =>  LabeledPoint( parts.last, Vectors.dense(parts.slice(0 , ncol - 2))))
})
testsLabeled.foreach(x => x.persist(StorageLevel.MEMORY_AND_DISK))
tests.foreach(x => x.unpersist())

/////// Get predictions for each out of bag sample
// Get vectors of predictions for each test dataset
val tmpList : List[org.apache.spark.rdd.RDD[(org.apache.spark.mllib.regression.LabeledPoint, Double)]]  = List()
val oobPlusPreds = tmpList.toBuffer
for ( b <- (0 to N_BOOTS-1) ){
    oobPlusPreds += testsLabeled(b).map( point => {
        val prediction = treeModels(b).predict(point.features)
       (point , prediction)
    })
}

// Merge all OOB's and the predictions
val stackedOOB = oobPlusPreds.reduce (   (a, b) => a.union(b)   )
val groupedOOB = stackedOOB.groupByKey
groupedOOB.persist(StorageLevel.MEMORY_AND_DISK)
trains.foreach(x => x.unpersist())


/* ------------------------------------------------ */
/* Take max vote (mode of  */
/* ------------------------------------------------ */

def mode( dat : List[Double] ) = {

    // Get list of form: (classification, #votes)
    val classCounts = dat.groupBy(x => x).mapValues(_.size).toList
    
    // Sort by the number of votes (descending)
    val classCountsSorted = classCounts.sortBy{_._2}.reverse
    
    // Find maximum number of votes
    val maxVotes = classCountsSorted(0)._2
    
    // Filter to only labels that got the max number of votes
    val modeCandidates = classCountsSorted.filter(x => x._2 == maxVotes)
    
    // Randomly select one as the mode
    val modeWinnerTuple = scala.util.Random.shuffle(modeCandidates).take(1)
    
    // Return the key, which is the label
    modeWinnerTuple(0)._1
}

val rfPredictions = groupedOOB.map(   line =>  {
    val predictions = line._2
    val prediction = mode(predictions.toList)
    (line._1.label, prediction)
})
rfPredictions.persist(StorageLevel.MEMORY_AND_DISK)

val oobError = rfPredictions.filter(r => r._1  != r._2).count.toDouble / rfPredictions.count

// End timer
val finish : Long = (Calendar.getInstance.getTime).getTime
val elapsedTime = (finish - start)/1000
/* ----------------------------------------------- */
/* Print results */
/* ----------------------------------------------- */

var timeStamp : String = s"${startTime}"
timeStamp = timeStamp.substring(4,19).replaceAll(" ", "_")

val file = new File(OUTDIR + timeStamp + ".txt" )
val bw = new BufferedWriter(new FileWriter(file))
bw.write(s"Error Rate         : ${oobError}\n")
bw.write(s"Elapsed Time       : ${elapsedTime} seconds\n")
bw.write(s"# Bootstraps       : ${N_BOOTS}\n")
bw.write(s"# Features Selected: ${N_FEATURES}\n")
bw.close()
