# Code review of Spark MLLib decision-tree PR 79 - Hirakendu Das

## General design notes

The current design of classes and relationships is good,
but I think it would be great if it can be modified slightly to
make it similar to the design of other existing algorithms in MLLib
and extend the existing interfaces.
For example, we can have a `DecisionTreeModel` or that extends
the existing `RegressionModel` interface, similar to
the existing `RidgeRegressionModel` and `LinearRegressionModel`.
Alternatively, we can have separate `ClassificationTreeModel`
and `RegressionTreeModel` that extend the existing interfaces
`ClassificationModel` and `RegressionModel` respectively.

Note that it is important to keep the `Model` as a class,
so that we can later compose ensemble and other composite models
consisting of multiple `Model` instances.
Currently the decision tree `Node` is essentially the `Model`,
although I would prefer a wrapper around it and explicitly
called `DecisionTreeModel` that implements methods like
`predict`, `save`, `explain`, `load` like I remember
coming across in MLI.

In the file `DecisionTree.scala`, the actual class
can be renamed to `DecisionTreeAlgorithm` that extends
`Algorithm` interface that implements `train(samples)`
and outputs a `RegressionModel`.
Note that there is no `Algorithm` interface currently in MLLib,
although there may be one in MLI and the closest is
`abstract class GeneralizedLinearAlgorithm[M <: GeneralizedLinearModel]`.

Modeling `Strategy` should be renamed to `Parameters`.
I would prefer a separation between model parameters
like depth, minimum gain and algorithm parameters
like level-by-level training or caching.
The line is blurred for some aspects like quantile strategy,
but I am inclined to put those into modeling parameters,
so it can be referred to later on.
Rule of thumb being that different algorithm parameters
should lead to same model (up to randomization effects)
for the same model parameters.

`Impurity` should be renamed to `Error` or something more technical.
Also see my later comments on the need for a generic `Error` Interface
that allow easy adaptation of algorithms for specific loss functions.

In general, MLLib and MLI should define proper interfaces
for `Model`, `Algorithm`, `Parameters` and importantly `Loss` entities,
at least for supervised machine learning algorithms
and all implementations should adhere to it.
Surprisingly, MLLIb or MLI currently doesn't have a `Loss` or `Error` interface,
the closest being the `Optimizer` interface.
There is also a need for portable model output formats, e.g., JSON,
that can be used by other programs, possibly written in other languages
outside Scala and Java.
Can also use an existing format like PMML (not sure if it's widely used).
Lastly, there is a need to support standard data formats with
optional delimiter parameters - I am sure this a general need for Spark.
I understand there has been significant effort before for standardization,
would be good to know about the current status.


## Error or Impurity Interface

Adding to the discussion on the need for a generic interface for `Impurity`,
or more precisely `Error`, I believe we all see that it's good to have.
Ideally I would have preferred a single `Error` trait and that all types
of Error like Square or KL divergence extend it, but the consensus is
that it negatively impacts performance.

In addition to performance-oriented implementations for specific loss functions,
I would still recommend a generic `Error` interface and a generic implementation
of decision-tree based on this interface.
One possibility is to add a third `calculate(stats)`,
or more precisely `error(errorStats: ErrorStats)` to the `Error` interface.
I am not sure it will help the signature collision problem though,
unless we just keep the one signature for generic error statistics.

For reference and example of one such interface and implementations, see
 `trait LossStats[S <: LossStats[S]]` and
`abstract class Loss[S <: LossStats[S]:Manifest]` in my previous PR,
[https://github.com/apache/incubator-spark/pull/161/files](https://github.com/apache/incubator-spark/pull/161/files),
that exactly do that and provide interfaces for
aggregable error statistics and calculating error from these statistics.
(On second thought, I feel `ErrorStats` and `Error` are better names.)
Also see the generic implementation
`class DecisionTreeAlgorithm[S <: LossStats[S]:Manifest]`
and implementations of specific error functions, `SquareLoss` and `EntropyLoss`.

## Miscellaneous notes

1. `DecisionTree` class and object should be reorganized and separated
   into `DecisionTreeModel` and `DecisionTreeAlgorithm`,
   with a `Strategy` and root `Node` as part of `Model` and
   `train` as part of the `Algorithm`.
   
2. `Filter` class is a nice abstraction of branching conditions leading
   to current node. There are already references to left and right child nodes,
   so I think this is redundant. If need be, a reference to a parent node
   as an `Option[Node]` suffices and is more useful. The functionality
   should be covered across `Node` and `Split` classes.

3. [Optional] The functionality of `Split` can be simplified by a modification.
   If I understand correctly, `Split` represents
   the left or right (low or high) branch of the parent node.
   Instead, it suffices to store the branching condition
   for each node as a splitting condition.
   This can be appropriately named as `SplitPredicate` or `SplittingCondition`
   or branching condition and consist of feature id,
   feature type (continuous or categorical), threshold,
   left branch categories and right branch categories.
   
   I think depending on the choice here, we require `Filter`, but
   nonetheless I think it's redundant and we should exploit
   the recursive/linked structure of tree, which we are doing anyway.

4. `Node` should be a recursive/linked structure with references to
   child nodes and parent node (the latter allows for easy traversal),
   so I don't see the need for `nodes: Array[Node]` as the model
   and the `build` method in `Node`. The `DecisionTreeModel` should
   essentially be the root `Node` with methods like `predict` etc.
   involving a recursive traversal on child nodes.
   
   The method `predictIfleaf` in `Node` should be renamed to simply
   `predict`. It predicts regardless of whether it's a leaf and does
   recursive traversal until it hits a leaf child.
   The prediction value should be renamed to `prediction` instead
   of `predict`, which would clean up the ambiguity with this `predict` method.
   
   Putting things together: `Node` should be simple with a `prediction`
   and should be a recursive structure. `DecisionTreeModel` should
   be a wrapper around a root `Node` member and contain methods like `predict`,
   `save`, `explain`, `load` etc. based on recursive traversal.

5. `Strategy` should be renamed to `Parameters`. Modeling and algorithm
   parameters can be separate, the latter being part of the model.

6. `Bin` class can be simplified and some members renamed.
   The `lowSplit`, `highSplit` can be simplified to the single
   threshold corresponding to the left end of the bin range.
   This can be named to `leftEnd` or `lowEnd`.
   
   [Optional] It's not clear this class is needed at first place.
   For categorical variables, the value itself is the bin index,
   and for continuous variables, bins are simply defined
   by candidate thresholds, in turn defined by quanties.
   For every feature id, one can maintain a list of categories
   and thresholds and be done. In that case, for continuous features,
   the position of the threshold is the bin index.

7. [Optional] `InformationGainStats` and `Split` nicely separate
   the members of `Node`, but can also be flattened and put at top level.
   Would make storage and explanation slightly easier, albeit less unstructured. 

8. `Impurity` should be renamed to `Error` or something more technical
   and familiar.
   Also see the comments earlier for the necessity and example design
   of a generic `Error` interface.
   
   The `calculate` method can be renamed to something verbose like `error`.
   
   For a generic interface, an additional `ErrorStats` trait
   and `error(errorStats: ErrorStats)` method can be added.
   For example, `Variance` or more aptly, `SquareError`, would implement
   `case class SquareErrorStats(count: Long, mean: Double, meanSquare: Double)`
   and `error(errorStats) = errorStats.meanSquare - errorStats.mean * errorStats.mean / count`.
   Note that `ErrorStats` should have aggregation methods, e.g., it's
   easy to see the implementation for `SquareErrorStats`.
   
   The `Variance` class should be renamed to `SquareError`,
   `Entropy` to `EntropyError` or `KLDivergence`, `Gini` to `GiniError`.

9. [Optional/Minor] The `Algorithm` Enumeration seems redundant due to `Impurity`,
   which decides the `Algorithm` anyway.

   The various `Enumeration` classes in `mllib.tree.configuration` package
   are neat. A uniform _design pattern_ for parameters and options should
   be used for MLLib and Spark, and this could be a start.
   Alternatively, if there is an existing pattern in use, it should
   be followed for decision tree as well.

10. Input data formats for main program should be made consistent
    with programs for other algorithms. Currently, it's CSV for decision-tree,
    but I believe it is `<label>,<feature1>\t<feature2>...` etc for
    other algorithms. I prefer a more _standard_ TSV format as used
    in Hadoop text format. Alternatively, if the label has to be separated,
    current _labeled point_ format used in other programs is fine.

11. [Optional/Minor] A plan should be made to have a consistent hierarchy
   and organization of various algorithms in the MLLib package.
   A separate `tree` subpackage seems unnecessary.

## Scalability and Error Performance Experiments

The code was tested on a dataset for a binary classification problem.
A regression tree with `Variance` (square loss) was trained because
it's computationally more intensive.
The dataset consists of about `500,000,000` training instances.
There are 20 features, all numeric. Although there are categorical features
in the dataset and the algorithm implementation supports it,
they were not used since the main program doesn't have options.

The dataset is of size about 90 GB in plain text format.
It consists of 100 part files, each about 900 MB.
To _optimize_ number of tasks to align with number of workers,
the task split size was chosen to be 160 MB to have 300 tasks.

The model training experiements were done on a Yahoo internal Hadoop cluster.
The CPUs are of type Intel Xeon 2.4 GHz.
The Spark YARN adaptation was used to run on Hadoop cluster.
Both master-memory and worker-memory were set to `7500m`
and the cluster was under light to moderate load at the time of experiments.
The fraction of memory used for caching was `0.3`,
leading to about `2 GB` of memory per worker for caching. 
The number of workers used for various experiments were
`20, 30, 60, 100, 150, 300, 600` to evenly align with 600 tasks.
Note that with 20 and 30 workers, only `48%` and `78%`
of the 90 GB data could be cached in memory.
The decision tree of depth 10 was trained and minor code changes
were made to record the individual level training times.
The training command (with additional JVM settings) used is

    time \
    SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} \
    -Dspark.hadoop.mapred.min.split.size=167772160 \
    -Dspark.hadoop.mapred.max.split.size=167772160" \
    -Dspark.storage.memoryFraction=0.3 \
    spark-class org.apache.spark.deploy.yarn.Client \
    --queue ${QUEUE} \
    --num-workers ${NUM_WORKERS} \
    --worker-memory 7500m \
    --worker-cores 1 \
    --master-memory 7500m \
    --jar ${JARS}/spark_mllib_tree.jar \
    --class org.apache.spark.mllib.tree.DecisionTree \
    --args yarn-standalone \
    --args --algo --args Regression \
    --args --trainDataDir --args ${DIST_WORK}/train_data_lp.txt \
    --args --testDataDir --args ${DIST_WORK}/test_data_0p1pc_lp.txt \
    --args --maxDepth --args 10 \
    --args --impurity --args Variance \
    --args --maxBins --args 100


The training times for training each depth and for each choice
of number of workers is in the attachments
[workers_times.txt](https://raw.githubusercontent.com/hirakendu/temp/master/spark_mllib_tree_pr79_review/workers_times.txt).
The attached graphs  demonstrate the scalability in terms of
cumulative training times for various depths and various number of workers.

![](https://raw.githubusercontent.com/hirakendu/temp/master/spark_mllib_tree_pr79_review/workers_times.png)

![](https://raw.githubusercontent.com/hirakendu/temp/master/spark_mllib_tree_pr79_review/workers_speedups.png)


For obtaining the speed-ups, the training times are compared to those
for 60 workers, since the data could not be cached completely for
20 and 30 workers. For all experiments, the resources requested
were fully allocated by the cluster and all experiments ran
to completion in their first and only run.
As we can see, the scaling is nearly linear for higher depths 9 and 10,
across the range of 60 workers to 600 workers,
although the slope is less than 1 as expected.
For such loads, 60 to 100 workers are a reasonable computing resource
and the total training times are about 160 minutes and 100 minutes respectively.
Overall, the performance is satisfactory, in particular when trees
of depth 4 or 5 are trained for boosting models.
But clearly, there is room for improvement in the following sense.
The dataset when fully cached takes about 10s for a count operation,
whereas the training time for first level that involves simple histogram
calculation of three error statistics takes roughly 30 seconds.

The error performance in terms of RMSE was verified to be close to
that of alternate implementations.

