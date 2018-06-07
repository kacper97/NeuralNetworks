package learning;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import javax.imageio.ImageIO;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import robocode.BattleResults;
import robocode.control.BattleSpecification;
import robocode.control.BattlefieldSpecification;
import robocode.control.RobocodeEngine;
import robocode.control.RobotSetup;
import robocode.control.RobotSpecification;
import robocode.control.events.BattleAdaptor;
import robocode.control.events.BattleCompletedEvent;
import robocode.control.events.BattleErrorEvent;
import robocode.control.events.BattleMessageEvent;

class Environment{
	static int NdxBattle;
	static final int NUMSAMPLES=1000;
	static double []FinalScore1 = new double[NUMSAMPLES];
	static double []FinalScore2 = new double[NUMSAMPLES];
	public static void main(String[] args) {
		BattlefieldParameterEvaluator bEvaluator = new BattlefieldParameterEvaluator(NUMSAMPLES,FinalScore1,FinalScore2);
		bEvaluator.run();
	}
}
public class BattlefieldParameterEvaluator {
	// Minimum allowable battlefield size is 400
	final int MAXBATTLEFIELDSIZE=4000;
	// Minimum allowable gun cooling rate is 0.1
	final double MAXGUNCOOLINGRATE=10;
	final int NUMBATTLEFIELDSIZES=601;
	final int NUMCOOLINGRATES=501;
	// Number of inputs for the multilayer perceptron (size of the input vectors) final static int NUM_NN_INPUTS=2;
	// Number of hidden neurons of the neural network
	final int NUM_NN_HIDDEN_UNITS=50;
	// Number of epochs for training
	final int NUM_TRAINING_EPOCHS=100000;
	int NdxBattle;
	int NUMSAMPLES=1000;
	double []FinalScore1 = new double[NUMSAMPLES];
	double []FinalScore2 = new double[NUMSAMPLES];
	double []BattlefieldSize=new double[NUMSAMPLES]; 
	double []GunCoolingRate=new double[NUMSAMPLES];
	
	public BattlefieldParameterEvaluator(int numSamples,double[] FS1, double[] FS2) {
		NUMSAMPLES=numSamples;
		FinalScore1 = FS1; 
		FinalScore2 = FS2;
	}

private void createDatasetForNN(double [][]RawInputs, double [][]RawOutputs) {
	for(int NdxSample=0;NdxSample<NUMSAMPLES;NdxSample++) {
		// TODO IMPORTANT: normalize the inputs and the outputs to
		// the interval [0,1] 
			RawInputs[NdxSample][0]=BattlefieldSize[NdxSample]/MAXBATTLEFIELDSIZE; 
			RawInputs[NdxSample][1]=GunCoolingRate[NdxSample]/MAXGUNCOOLINGRATE; 
			RawOutputs[NdxSample][0]=FinalScore1[NdxSample]/250;
		}
}

private void createAndTrainNN(double [][]RawInputs, double [][]RawOutputs, int numOfEpochs) {
	BasicNeuralDataSet MyDataSet=new BasicNeuralDataSet(RawInputs,RawOutputs);
	
	BasicNetwork network = new BasicNetwork();
	int hiddenUnitsCout = 50;
	network.addLayer(new BasicLayer(new ActivationSigmoid(),false,hiddenUnitsCout)); // No need for bias
	network.getStructure().finalizeStructure();
	network.reset();
	
	
	// training
	ResilientPropagation propagation = new ResilientPropagation(network, MyDataSet);
	for (int i = 0; i < numOfEpochs; i++) {
		propagation.iteration();
	}
	
	// TODO run tests
}

private void buildOutputImage(int numInputs) {
	// Generate test samples to build an output image
	int []OutputRGBint=new int[NUMBATTLEFIELDSIZES*NUMCOOLINGRATES]; 
	double MyValue=0;
	double [][]MyTestData=new double [NUMBATTLEFIELDSIZES*NUMCOOLINGRATES][numInputs];
	
	
	for(int NdxBattleSize=0;NdxBattleSize<NUMBATTLEFIELDSIZES;NdxBattleSize++) {
		for(int NdxCooling=0;NdxCooling<NUMCOOLINGRATES;NdxCooling++) {
			MyTestData[NdxCooling+NdxBattleSize*NUMCOOLINGRATES][0]= 0.1+0.9*((double)NdxBattleSize)/NUMBATTLEFIELDSIZES;
			MyTestData[NdxCooling+NdxBattleSize*NUMCOOLINGRATES][1]= 0.1+0.9*((double)NdxCooling)/NUMCOOLINGRATES;
		} 
	}
	
	// Simulate the neural network with the test samples and fill a matrix
	for(int NdxBattleSize=0;NdxBattleSize<NUMBATTLEFIELDSIZES;NdxBattleSize++) {
		for(int NdxCooling=0;NdxCooling<NUMCOOLINGRATES;NdxCooling++) {
			double MyResult= MyTestData[NdxCooling+NdxBattleSize*NUMCOOLINGRATES][1]; // TODO what is this?
			MyValue=ClipColor(MyResult); 
			Color MyColor = new Color((float)MyValue,
									(float)MyValue,
									(float)MyValue); 
			OutputRGBint[NdxCooling+NdxBattleSize*NUMCOOLINGRATES]=MyColor.getRGB(); 
		}
	}
	
	System.out.println("Testing completed.");
	
	// Plot the training samples
	for(int NdxSample=0;NdxSample<NUMSAMPLES;NdxSample++) {
		MyValue=ClipColor(FinalScore1[NdxSample]/250); 
		Color MyColor=new Color((float)MyValue,(float)MyValue,(float)MyValue); 
		int MyPixelIndex = (int)(Math.round(NUMCOOLINGRATES*((GunCoolingRate[NdxSample]/MAXGUNCOOLINGRATE)-0.1)/0.9)+
								Math.round(NUMBATTLEFIELDSIZES*((BattlefieldSize[NdxSample] /MAXBATTLEFIELDSIZE)-0.1)/0.9)*NUMCOOLINGRATES);
		if ((MyPixelIndex>=0) && (MyPixelIndex<NUMCOOLINGRATES*NUMBATTLEFIELDSIZES))
		{
			OutputRGBint[MyPixelIndex]=MyColor.getRGB();
		} 
	}
	
	BufferedImage img=new BufferedImage (NUMCOOLINGRATES,NUMBATTLEFIELDSIZES,BufferedImage.TYPE_INT_RGB);
	img.setRGB(0, 0, NUMCOOLINGRATES, NUMBATTLEFIELDSIZES, OutputRGBint, 0, NUMCOOLINGRATES);
	File f=new File("hello.png"); 
	try {
	ImageIO.write(img,"png",f); 
	} catch (IOException e) {
	// TODO Auto‐generated catch block 
		e.printStackTrace();
	}
	System.out.println("Image generated.");
}
public void run() {
	runBattles();
	
	// Create the training dataset for the neural network
	final int NUM_NN_INPUTS = 50; // TODO adjust size
	final int NUM_OF_EPOCHS = 1;
	double [][]RawInputs=new double[NUMSAMPLES][NUM_NN_INPUTS]; 
	double [][]RawOutputs=new double[NUMSAMPLES][1];
	
	createDatasetForNN(RawInputs,RawOutputs);
	
	createAndTrainNN(RawInputs, RawOutputs, NUM_OF_EPOCHS);
	
	System.out.println("Training network...");
	System.out.println("Training completed.");
	System.out.println("Testing network...");
	
	buildOutputImage(NUM_NN_INPUTS);
		
	// Make sure that the Java VM is shut down properly
	System.exit(0);
}
/*
 * Clip a color value (double precision) to lie in the valid range [0,1]
 */
public static double ClipColor(double Value) {
	if (Value<0.0) {Value=0.0;}
	else if (Value>1.0){Value=1.0;}
	return Value;
}

//
// Our private battle listener for handling the battle event we are interested in. 
//
static class BattleObserver extends BattleAdaptor {
	// Called when the battle is completed successfully with battle results
	public void onBattleCompleted(BattleCompletedEvent e) { 
		System.out.println("‐‐ Battle has completed ‐‐");
		
	    // Get the indexed battle results
	    BattleResults []results=e.getIndexedResults();
	    
	    // Print out the indexed results with the robot names
	    System.out.println("Battle results:"); 
	    for (BattleResults result : results) {
	    		System.out.println(" " + result.getTeamLeaderName() + ": " + result.getScore());
	    }
	// Store the scores of the robots
		Environment.FinalScore1[Environment.NdxBattle] = results[0].getScore(); 
		Environment.FinalScore2[Environment.NdxBattle] = results[1].getScore();
	 }
	}
	// Called when the game sends out an information message during the battle
	public void onBattleMessage(BattleMessageEvent e) { 
		//System.out.println("Msg> " + e.getMessage());
	}
	// Called when the game sends out an error message during the battle
	public void onBattleError(BattleErrorEvent e) { 
		System.out.println("Err> " + e.getError());
	} 

	private void runBattles() {
		Random rng=new Random(15L);
		
		// Disable log messages from Robocode
		RobocodeEngine.setLogMessagesEnabled(false);
		
		// Create the RobocodeEngine
		RobocodeEngine engine = new RobocodeEngine(new java.io.File("C:/Robocode"));
		//RobocodeEngine engine = new RobocodeEngine(new java.io.File("/Users/denisdrobny/robocode")); // TODO
		// Add our own battle listener to the RobocodeEngine
		engine.addBattleListener(new BattleObserver()); 
		
		// Show the Robocode battle view
		engine.setVisible(false);
		
		// Setup the battle specification
		// Setup battle parameters
		int numberOfRounds = 1; 
		long inactivityTime = 100; 
		int sentryBorderSize = 50;
		boolean hideEnemyNames = false;
		
		// Get the robots and set up their initial states
		RobotSpecification[] competingRobots =
		        engine.getLocalRepository("sample.RamFire,sample.TrackFire");
		RobotSetup[] robotSetups = new RobotSetup[2]; 
		for(NdxBattle=0;NdxBattle<NUMSAMPLES;NdxBattle++)
		 {
			// Choose the battlefield size and gun cooling rate
			BattlefieldSize[NdxBattle]= MAXBATTLEFIELDSIZE*(0.1+0.9*rng.nextDouble());
			GunCoolingRate[NdxBattle]=MAXGUNCOOLINGRATE*(0.1+0.9*rng.nextDouble());
			
			// Create the battlefield
			BattlefieldSpecification battlefield =
			new BattlefieldSpecification((int)BattlefieldSize[NdxBattle], (int)BattlefieldSize[NdxBattle]);
			// Set the robot positions
			robotSetups[0]=new RobotSetup(BattlefieldSize[NdxBattle]/2.0, BattlefieldSize[NdxBattle]/3.0,0.0);
			robotSetups[1]=new RobotSetup(BattlefieldSize[NdxBattle]/2.0, 2.0*BattlefieldSize[NdxBattle]/3.0,0.0);
			// Prepare the battle specification
			BattleSpecification battleSpec =
			new BattleSpecification(battlefield,
									numberOfRounds, 
									inactivityTime, 
									GunCoolingRate[NdxBattle], 
									sentryBorderSize, 
									hideEnemyNames, 
									competingRobots, 
									robotSetups);
			
			// Run our specified battle and let it run till it is over 
			engine.runBattle(battleSpec, true); // waits till the battle finishes
		}
		
		// Cleanup our RobocodeEngine
		engine.close();
		
		// Show results
		System.out.println(Arrays.toString(BattlefieldSize)); 
		System.out.println(Arrays.toString(GunCoolingRate)); 
		System.out.println(Arrays.toString(FinalScore1)); 
		System.out.println(Arrays.toString(FinalScore2));
	}
}





