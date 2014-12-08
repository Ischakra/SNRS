package crowdsourcing;
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import crowdsourcing.AdjCosineSimilarity;

import com.google.common.collect.Iterables
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.LossBalancingType
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.FullInferenceResult
import edu.umd.cs.psl.evaluation.statistics.ContinuousPredictionComparator
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries



def dataPath = "./data/yelp/"

Logger log = LoggerFactory.getLogger(this.class)
ConfigManager cm = ConfigManager.getManager();
ConfigBundle cb = cm.getBundle("crowdsourcing");

def defPath = System.getProperty("java.io.tmpdir") + "/psl-yelp"
def dbpath = cb.getString("dbpath", defPath)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), cb)

ExperimentConfigGenerator configGenerator = new ExperimentConfigGenerator("Yelp");
methods = ["MLE"];
configGenerator.setLearningMethods(methods);
configGenerator.setVotedPerceptronStepCounts([100]);
configGenerator.setVotedPerceptronStepSizes([(double) 1.0]);

/*** MODEL DEFINITION ***/

log.info("Initializing model ...");

PSLModel m = new PSLModel(this, data);

m.add predicate: "user" , types: [ArgumentType.UniqueID]
m.add predicate: "bussiness" , types: [ArgumentType.UniqueID]
m.add predicate: "rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "ratingFriendsMajority", types: [ArgumentType.UniqueID]//need to think about its value and arguments I think /*
m.add predicate: "ratingPrior", types [ArgumentType.UniqueID]
m.add predicate: "friends" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "review_count" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "bestReviewer" , types: [ArgumentType.UniqueID]
m.add predicate: "votes", types, types : [ArgumentType.UniqueID
m.add predicate: "fans" , types: [ArgumentType.UniqueID]

m.add predicate: "similarUser", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "similarItem", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]


m.add function: "similaruser" , implementation: new userSimilarity()*/write implementation*/
m.add function: "similaritem" , implementation: new itemSimilarity ()
m.add function: "ratingFriendsMajority", implementation : new ratingFriendsMajority()// majority rating among friends/*
m.add function: "bestReviewer" , implementation : new bestReviewer()

//two sided prior
UniqueID constant = data.getUniqueID(0)
m.add rule: ( user(U) & joke(J) & ratingPrior(constant) ) >> rating(U,J), weight: 5.0, squared: "Y";
m.add rule: ( rating(U,J) ) >> ratingPrior(constant), weight: 5.0, squared: sq;

m.add rule : ( user(u) & bussiness(B) & ratingFriendsMajority(B)) >> rating (u,B) , weight : 5
m.add rule : (friends(u1,u2) & rating (u1,B) >> rating (u2,B) , weight :5
m.add rule : ( friends(u1,u2) & similarUser(u1,u2) & rating (u1,B)) >> rating (u2,B) , weight : 5
m.add rule : ( friends(u1,u2) & bestReviewer(u1) & rating (u1,B))>> rating (u2,B) , weight :5
m.add rule : ( similarItem(B1,B2) & rating (u,B1)) >> rating (u, B2), weight :5
m.add rule : ( similaruser (u1,u2) & rating (u1,B)) >> rating (u2,B) , weight :5

log.info("Model: {}", m)

/* get all default weights */
Map<CompatibilityKernel,Weight> initWeights = new HashMap<CompatibilityKernel, Weight>()
for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
	initWeights.put(k, k.getWeight());

ExperimentConfigGenerator configGenerator = new ExperimentConfigGenerator("yelp");

methods = ["MLE"];
configGenerator.setLearningMethods(methods);

/* MLE/MPLE options */
configGenerator.setVotedPerceptronStepCounts([100]);
configGenerator.setVotedPerceptronStepSizes([(double) 1.0]);
folds = 10

// assigning configbundle to the experiment, creating maps to keep results
List<ConfigBundle> configs = configGenerator.getConfigs();
Map<ConfigBundle,ArrayList<Double>> expResults = new HashMap<String,ArrayList<Double>>();
for (ConfigBundle config : configs) {
	expResults.put(config, new ArrayList<Double>(folds));
}
// insert data in training and test partitions, each of which has read, write and labels(i.e. given values to compare with inferred values) 
for (int fold = 0; fold < folds; fold++) {

	Partition read_tr = new Partition(0 + fold * folds);
	Partition write_tr = new Partition(1 + fold * folds);
	Partition read_te = new Partition(2 + fold * folds);
	Partition write_te = new Partition(3 + fold * folds);
	Partition labels_tr = new Partition(4 + fold * folds);
	Partition labels_te = new Partition(5 + fold * folds);

	def inserter;
	// users
	inserter = data.getInserter(user, read_tr);
	InserterUtils.loadDelimitedData(inserter, dataPath + "/users-tr-1000.txt");// need to change
	inserter = data.getInserter(user, read_te);
	InserterUtils.loadDelimitedData(inserter, dataPath + "/users-te-1000.txt");// need to change
	// businesses
	inserter = data.getInserter(business, read_tr);
	InserterUtils.loadDelimitedData(inserter, dataPath + "/business.txt");// need to put txt files
	inserter = data.getInserter(business, read_te);
	InserterUtils.loadDelimitedData(inserter, dataPath + "/business.txt");
	//user-user cosine similarity by rating
	inserter = data.getInserter(similarUser, read_tr);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/userSimilarity.txt");
	inserter = data.getInserter(similarUser, read_te);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/userSimilarity.txt");
	// review text similarity
	inserter = data.getInserter(simJokeText, read_tr);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "//ReviewTextSim.txt");//
	inserter = data.getInserter(simJokeText, read_te);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/ReviewTextSim.txt");
	// observed ratings
	inserter = data.getInserter(rating, read_tr);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/bussiness/ratings-tr-obs-" + fold + ".txt");
	inserter = data.getInserter(rating, read_te);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/bussiness/ratings-te-obs" + fold + ".txt");
	//I have not kept ratingObs
	// unobserved ratings (ground truth)
	inserter = data.getInserter(rating, labels_tr);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/bussiness/ratings-tr-uno-" + fold + ".txt");
	inserter = data.getInserter(rating, labels_te);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/bussiness/ratings-te-uno-" + fold + ".txt");
	// prior (we'll overwrite later) need to modify
	data.getInserter(ratingPrior, read_tr).insertValue(0.5, constant)
	data.getInserter(ratingPrior, read_te).insertValue(0.5, constant)
	
	/** POPULATE DB ***/

	/* We want to populate the database with all groundings 'rating' and 'ratingObs'
	 * To do so, we will query for all users and jokes in train/test, then use the
	 * database populator to compute the cross-product. 
	 */
	DatabasePopulator dbPop;
	Variable User = new Variable("User");
	Variable Joke = new Variable("Business");
	Set<GroundTerm> users = new HashSet<GroundTerm>();
	Set<GroundTerm> business = new HashSet<GroundTerm>();
/* what are the following lines doing?
Map<Variable, Set<GroundTerm>> subs = new HashMap<Variable, Set<GroundTerm>>();
	subs.put(User, users);
	subs.put(Joke, jokes);*/
	ResultList results;
	def toClose;
	
	AdjCosineSimilarity userCosSim = new AdjCosineSimilarity(rating, 1, avgJokeRatingObs, simThresh);// check
	AdjCosineSimilarity bussinessCosSim = new AdjCosineSimilarity(rating, 0, avgUserRatingObs, simThresh);//check
	
	
	
