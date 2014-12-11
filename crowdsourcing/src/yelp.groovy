package crowdsourcing;
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import crowdsourcing.AdjCosineSimilarity;
import util.ExperimentConfigGenerator;
import edu.umd.cs.bachuai13.util.WeightLearner;

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



def dataPath = "./data/"

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
//m.add predicate: "ratingFriendsMajority", types: [ArgumentType.UniqueID]//need to think about its value and arguments I think /*
m.add predicate: "ratingPrior", types [ArgumentType.UniqueID]
m.add predicate: "businessAvgRating", types [ArgumentType.UniqueID]
m.add predicate: "friends" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "review_count" , types: [ArgumentType.UniqueID]
m.add predicate: "bestReviewer" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
//m.add predicate: "votes", types, types : [ArgumentType.UniqueID//
m.add predicate: "fans" , types: [ArgumentType.UniqueID]

m.add predicate: "similarUser", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "similarItem", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]


m.add function: "similaruser" , implementation: new userSimilarity()*/write implementation*/
m.add function: "similaritem" , implementation: new itemSimilarity ()
m.add function: "ratingFriendsMajority", implementation : new ratingFriendsMajority()// majority rating among friends/*
m.add function: "bestReviewer" , implementation : new bestReviewer()

//two sided prior
UniqueID constant = data.getUniqueID(0)
m.add rule: ( user(U) & business(J) & ratingPrior(U) ) >> rating(U,J), weight: 5.0;
m.add rule: ( rating(U,J) ) >> ratingPrior(U), weight: 5.0 ;

//m.add rule : ( user(u) & business(B) & ratingFriendsMajority(B)) >> rating (u,B) , weight : 5
m.add rule : ( friends(u1,u2) & rating (u1,B) ) >> rating (u2,B) , weight :5;
m.add rule : ( friends(u1,u2) & similarUser(u1,u2) & rating (u1,B)) >> rating (u2,B) , weight : 5
//m.add rule : ( friends(u1,u2) & bestReviewer(u1,u2) & rating (u1,B))>> rating (u2,B) , weight :5
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
folds = 1
simThresh= 0.5

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
	InserterUtils.loadDelimitedData(inserter, dataPath + "/users-tr.txt");// need to change
	inserter = data.getInserter(user, read_te);
	InserterUtils.loadDelimitedData(inserter, dataPath + "/users-te.txt");// need to change
	// businesses
	inserter = data.getInserter(business, read_tr);
	InserterUtils.loadDelimitedData(inserter, dataPath + "/MLN-businesses.txt");// need to put txt files
	inserter = data.getInserter(business, read_te);
	InserterUtils.loadDelimitedData(inserter, dataPath + "/ MLN-businesses.txt");
	//Friends
	inserter = data.getInserter(friends, read_tr);
	InserterUtils.loadDelimitedData(inserter, dataPath + "/MLN-friends.txt");// need to put txt files
	inserter = data.getInserter(friends, read_te);
	InserterUtils.loadDelimitedDatatruth(inserter, dataPath + "/ MLN-friends.txt");
	//user-user cosine similarity by rating
//	inserter = data.getInserter(similarUser, read_tr);
//	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/userSimilarity.txt");
//	inserter = data.getInserter(similarUser, read_te);
//	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/userSimilarity.txt");
	// review text similarity
//	inserter = data.getInserter(simJokeText, read_tr);
//	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "//ReviewTextSim.txt");//
//	inserter = data.getInserter(simJokeText, read_te);
//	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/ReviewTextSim.txt");
	// observed ratings
	inserter = data.getInserter(rating, read_tr);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/ratings/yelp-1-tr-obs-" + fold + ".txt");
	inserter = data.getInserter(rating, read_te);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/ratings/yelp-1-te-obs" + fold + ".txt");
	//I have not kept ratingObs
	// unobserved ratings (ground truth)
	inserter = data.getInserter(rating, labels_tr);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/ratings/yelp-1-tr-uno-" + fold + ".txt");
	inserter = data.getInserter(rating, labels_te);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "/ratings/yelp-1-te-uno-" + fold + ".txt");
	// prior (we'll overwrite later) need to modify
//	data.getInserter(ratingPrior, read_tr).insertValue(0.5, constant)
//	data.getInserter(ratingPrior, read_te).insertValue(0.5, constant)
	inserter = data.getInserter(ratingPrior, read_tr);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "MLN-user-avg-rating.txt" + fold + ".txt");
	inserter = data.getInserter(ratingPrior, read_te);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "MLN-user-avg-rating.txt" + fold + ".txt");
	//businessAvgRatings
	inserter = data.getInserter(businessAvgRating, read_tr);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "MLN-business-avg-rating.txt" + fold + ".txt");
	inserter = data.getInserter(businessAvgRating, read_te);
	InserterUtils.loadDelimitedDataTruth(inserter, dataPath + "MLN-business-avg-rating.txt" + fold + ".txt");

	/** POPULATE DB ***/

	/* We want to populate the database with all groundings 'rating' 
	 * To do so, we will query for all users and bussiness in train/test, then use the
	 * database populator to compute the cross-product. 
	 
	 **********check **************************************
	 */
	DatabasePopulator dbPop;
	Variable User = new Variable("User");
	Variable Business = new Variable("Business");
	Set<GroundTerm> users = new HashSet<GroundTerm>();
	Set<GroundTerm> business = new HashSet<GroundTerm>();
	Map<Variable, Set<GroundTerm>> subs = new HashMap<Variable, Set<GroundTerm>>();
	subs.put(User, users);
	subs.put(Business, business);
	def toClose;
 	AdjCosineSimilarity userCosSim = new AdjCosineSimilarity(rating, 1, ratingPrior, simThresh);
	AdjCosineSimilarity businessCosSim = new AdjCosineSimilarity(rating, 0, businessAvgRating, simThresh);
	
	Database trainDB = data.getDatabase(read_tr);
	ResultList userGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(user));
	
	 for (int i = 0; i < userGroundings.size(); i++) {
		GroundTerm u = userGroundings.get(i)[0];
		users.add(u);// adding u to the Map of users
		// need not calcul ate avg
	/*	RandomVariableAtom a = (RandomVariableAtom) trainDB.getAtom(avgUserRating, u)// change the avg
		a.setValue(avg);// need to read given avg values from file before this
		trainDB.commit(a); */
	}
	
	ResultList businessGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(business));
	for (int i = 0; i < businessGroundings.size(); i++) {
		GroundTerm b = businessGroundings.get(i)[0];
		business.add(b);// adding b to the Map of business
		// need not calculate avg
	//	RandomVariableAtom a = (RandomVariableAtom) trainDB.getAtom(avgBusinessRating, b)// change the avg
	//	a.setValue(avg);// need to read given avg values from file before this
	//	trainDB.commit(a);
	}
	
	
	/* Precompute the similarities of users and businesses */
	log.info("Computing training business similarities ...")
	int nnzSim = 0;
	double avgsim = 0.0;
//	PrintWriter writer = new PrintWriter("the-file-name.txt", "UTF-8");
//	writer.println("The first line");
	List<GroundTerm> businessList = new ArrayList(business);
	for (int i = 0; i < businessList.size(); i++) {
		GroundTerm j1 = businessList.get(i);
		for (int j = i+1; j < businessList.size(); j++) {
			GroundTerm j2 = businessList.get(j);
			double s = businessCosSim.getValue(trainDB, j1, j2);
			log.info(" business similarity : {} " s)
			// write a text file here
			system.out.println ()
			if (s > 0.0) {
				/* upper half */ // why half ?
				RandomVariableAtom a = (RandomVariableAtom) trainDB.getAtom(simObsRatingB, j1, j2);
				a.setValue(s);
				trainDB.commit(a);
				/* lower half */
				a = (RandomVariableAtom) trainDB.getAtom(simObsRatingB, j2, j1);
				a.setValue(s);
				trainDB.commit(a);
				/* update stats */
				++nnzSim;
				avgsim += s;
			}
		}
	}
	//write.close
	log.info("Computing training user similarities ...")
	int nnzSim = 0;
	double avgsim = 0.0;
	List<GroundTerm> usersList = new ArrayList(users);
	for (int i = 0; i < usersList.size(); i++) {
		GroundTerm j1 = usersList.get(i);
		for (int j = i+1; j < usersList.size(); j++) {
			GroundTerm j2 = usersList.get(j);
			double s = userCosSim.getValue(trainDB, j1, j2);
			log.info (" user similarity : {} ", s)
			if (s > 0.0) {
				/* upper half */ // why half ?
				RandomVariableAtom a = (RandomVariableAtom) trainDB.getAtom(simObsRatingB, j1, j2);
				a.setValue(s);
				trainDB.commit(a);
				/* lower half */
				a = (RandomVariableAtom) trainDB.getAtom(simObsRatingB, j2, j1);
				a.setValue(s);
				trainDB.commit(a);
				/* update stats */
				++nnzSim;
				avgsim += s;
			}
		}
	}
	
	log.info("  Number nonzero sim (train): {}", nnzSim);// check
	log.info("  Average joke rating sim (train): {}", avgsim / nnzSim);// check whether twice for user & business
	trainDB.close();
	
	
	log.info("Populating training database ...");
	toClose = [user,business,ratingObs,ratingPrior,simJokeText,avgUserRatingObs,avgJokeRatingObs,simObsRating] as Set;// check
	trainDB = data.getDatabase(write_tr, toClose, read_tr);
	dbPop = new DatabasePopulator(trainDB);
	dbPop.populate(new QueryAtom(rating, User, Business), subs);
	Database labelsDB = data.getDatabase(labels_tr, [rating] as Set)

	/* Clear the users, business so we can reuse */
	users.clear();
	business.clear();
	/* Get the test set users/business
	 *
	 */
	Database testDB = data.getDatabase(read_te);
	ResultList userGroundings = testDB.executeQuery(Queries.getQueryForAllAtoms(user));
	for (int i = 0; i < userGroundings.size(); i++) {
		GroundTerm u = userGroundings.get(i)[0];
		users.add(u);// adding u to the Map of users
		// need not calculate avg
	//	RandomVariableAtom a = (RandomVariableAtom) testDB.getAtom(avgUserRating, u)// change the avg
	//	a.setValue(avg);// need to read given avg values from file before this
	//	testDB.commit(a);
	}
	
	ResultList businessGroundings = testDB.executeQuery(Queries.getQueryForAllAtoms(business));
	for (int i = 0; i < businessGroundings.size(); i++) {
		GroundTerm b = businessGroundings.get(i)[0];
		business.add(b);// adding b to the Map of business
		// need not calculate avg
	//	RandomVariableAtom a = (RandomVariableAtom) testDB.getAtom(avgBusinessRating, b)// change the avg
	//	a.setValue(avg);// need to read given avg values from file before this
	//	testDB.commit(a);
	}
// did not compute the priors, need to do something

	/* Precompute the similarities of users and businesses */
	log.info("Computing training business similarities ...")
	int nnzSim = 0;
	double avgsim = 0.0;
	List<GroundTerm> businessList = new ArrayList(business);
	for (int i = 0; i < businessList.size(); i++) {
		GroundTerm j1 = businessList.get(i);
		for (int j = i+1; j < businessList.size(); j++) {
			GroundTerm j2 = businessList.get(j);
			double s = businessCosSim.getValue(testDB, j1, j2);
			if (s > 0.0) {
				/* upper half */ // why half ?
				RandomVariableAtom a = (RandomVariableAtom) testDB.getAtom(simObsRatingB, j1, j2);
				a.setValue(s);
				testDB.commit(a);
				/* lower half */
				a = (RandomVariableAtom) testDB.getAtom(simObsRatingB, j2, j1);
				a.setValue(s);
				testDB.commit(a);
				/* update stats */
				++nnzSim;
				avgsim += s;
			}
		}
	}
	
	log.info("Computing training user similarities ...")
	int nnzSim = 0;
	double avgsim = 0.0;
	List<GroundTerm> usersList = new ArrayList(users);
	for (int i = 0; i < usersList.size(); i++) {
		GroundTerm j1 = usersList.get(i);
		for (int j = i+1; j < usersList.size(); j++) {
			GroundTerm j2 = usersList.get(j);
			double s = userCosSim.getValue(testDB, j1, j2);
			if (s > 0.0) {
				/* upper half */ // why half ?
				RandomVariableAtom a = (RandomVariableAtom) testDB.getAtom(simObsRatingB, j1, j2);
				a.setValue(s);
				testDB.commit(a);
				/* lower half */
				a = (RandomVariableAtom) testDB.getAtom(simObsRatingB, j2, j1);
				a.setValue(s);
				testDB.commit(a);
				/* update stats */
				++nnzSim;
				avgsim += s;
			}
		}
	}
	
	log.info("  Number nonzero sim (train): {}", nnzSim);// check
	log.info("  Average business rating sim (train): {}", avgsim / nnzSim);// check whether twice for user & business
	testDB.close();
	/* Populate testing database. */
	log.info("Populating testing database ...");
	toClose = [user,business,rating,ratingPrior,avgBusinessRating,simObsRating] as Set;// check with sachi
	testDB = data.getDatabase(write_te, toClose, read_te);
	dbPop = new DatabasePopulator(testDB);
	dbPop.populate(new QueryAtom(rating, User, Business), subs);
	// no labels?
	testDB.close();
	
	
	/*** EXPERIMENT ***////check
	log.info("Starting experiment ...");
	for (int configIndex = 0; configIndex < configs.size(); configIndex++) {
		ConfigBundle config = configs.get(configIndex);
		def configName = config.getString("name", "");
		def method = config.getString("learningmethod", "");

		/* Weight learning */
		WeightLearner.learn(method, m, trainDB, labelsDB, initWeights, config, log)

		log.info("Learned model {}: \n {}", configName, m.toString())

		/* Inference on test set */
		Database predDB = data.getDatabase(write_te, toClose, read_te);
		Set<GroundAtom> allAtoms = Queries.getAllAtoms(predDB, rating)
		for (RandomVariableAtom atom : Iterables.filter(allAtoms, RandomVariableAtom))
			atom.setValue(0.0)
			/* For discrete MRFs, "MPE" inference will actually perform marginal inference */
		MPEInference mpe = new MPEInference(m, predDB, config)
		FullInferenceResult result = mpe.mpeInference()
		log.info("Objective: {}", result.getTotalWeightedIncompatibility())
		predDB.close();
		
		/* Evaluation *///check
		predDB = data.getDatabase(write_te);
		Database groundTruthDB = data.getDatabase(labels_te, [rating] as Set)
		def comparator = new ContinuousPredictionComparator(predDB)
		comparator.setBaseline(groundTruthDB)
		def metrics = [ContinuousPredictionComparator.Metric.MSE, ContinuousPredictionComparator.Metric.MAE]
		double [] score = new double[metrics.size()]
		for (int i = 0; i < metrics.size(); i++) {
			comparator.setMetric(metrics.get(i))
			score[i] = comparator.compare(rating)
		}
		log.info("Fold {} : {} : MSE {} : MAE {}", fold, configName, score[0], score[1]);
		expResults.get(config).add(fold, score);
		predDB.close();
		groundTruthDB.close()
	}
	trainDB.close()
}
	
log.info("\n\nRESULTS\n");
for (ConfigBundle config : configs) {
	def configName = config.getString("name", "")
	def scores = expResults.get(config);
	for (int fold = 0; fold < folds; fold++) {
		def score = scores.get(fold)
		log.info("{} \t{}\t{}\t{}", configName, fold, score[0], score[1]);
		log.Debug("{} \t{}\t{}\t{}", configName, fold, score[0], score[1]);// added
	}
}
