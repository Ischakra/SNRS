package crowdsourcing;

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


log.info("Initializing model ...");

PSLModel m = new PSLModel(this, data);

m.add predicate: "user" , types: [ArgumentType.UniqueID]
m.add predicate: "bussiness" , types: [ArgumentType.UniqueID]
m.add predicate: "rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "ratingFriendsMajority", types: [ArgumentType.UniqueID]*/need to think about its value and arguments I think /*
m.add predicate: "ratingPrior", types [ArgumentType.UniqueID]
m.add predicate: "friends" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "review_count" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "bestReviewer" , types: [ArgumentType.UniqueID]
m.add predicate: "votes", types
m.add predicate: "fans" , types: [ArgumentType.UniqueID]

m.add predicate: "similarUser", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "similarItem", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]


m.add function: "similaruser" , implementation: new userSimilarity()*/write implementation*/
m.add function: "similaritem" , implementation: new itemSimilarity ()
m.add function: "ratingFriendsMajority", implementation : new ratingFriendsMajority()*/ majority rating among friends/*
m.add function: "bestReviewer" , implementation : new bestReviewer()

//two sided prior
UniqueID constant = data.getUniqueID(0)
m.add rule: ( user(U) & joke(J) & ratingPrior(constant) ) >> rating(U,J), weight: 5.0, squared: sq;
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

methods = "MLE";
configGenerator.setLearningMethods(methods);

/* MLE/MPLE options */
configGenerator.setVotedPerceptronStepCounts([100]);
configGenerator.setVotedPerceptronStepSizes([(double) 1.0]);
