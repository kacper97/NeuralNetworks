package learning;
import java.util.Arrays;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.beans.beancontext.BeanContext;
import java.io.File;
import java.io.IOException;
import java.util.Random; 
import javax.imageio.ImageIO;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.data.basic.BasicNeuralData;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.simple.EncogUtility;

import robocode.BattleResults; 
import robocode.control.*;
import robocode.control.events.*;

class Environment{
	static int NdxBattle;
	static final int NUMSAMPLES=1000; // TODO 1000
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
	int NUMSAMPLES;
	double []FinalScore1;
	double []FinalScore2;
	double []BattlefieldSize;
	double []GunCoolingRate;
	
	public BattlefieldParameterEvaluator(int numSamples,double[] FS1, double[] FS2) {
		NUMSAMPLES=numSamples;
		FinalScore1 = FS1; 
		FinalScore2 = FS2;
		BattlefieldSize = new double[NUMSAMPLES]; 
		GunCoolingRate = new double[NUMSAMPLES];
	}

	private void createDatasetForNN(double [][]RawInputs, double [][]RawOutputs) {
		for(int NdxSample=0;NdxSample<NUMSAMPLES;NdxSample++) {
			// IMPORTANT: normalize the inputs and the outputs to
			// the interval [0,1] 
				RawInputs[NdxSample][0]=BattlefieldSize[NdxSample]/MAXBATTLEFIELDSIZE; 
				RawInputs[NdxSample][1]=GunCoolingRate[NdxSample]/MAXGUNCOOLINGRATE; 
				// TODO other parameters
				RawOutputs[NdxSample][0]=FinalScore1[NdxSample]/250; 
			}
	}

	private BasicNetwork createAndTrainNN(double [][]RawInputs, double [][]RawOutputs, int numOfEpochs) {
		BasicNeuralDataSet MyDataSet=new BasicNeuralDataSet(RawInputs,RawOutputs);
		// Create training network
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null,true,1)) ;
		network.addLayer(new BasicLayer(new ActivationSigmoid(),false,NUM_NN_HIDDEN_UNITS)); // No need for bias
		network.addLayer(new BasicLayer(new ActivationSigmoid(),false,1)); 
		network.getStructure().finalizeStructure();
		network.reset();
		
		
		// Train
		ResilientPropagation propagation = new ResilientPropagation(network, MyDataSet);
		for (int i = 0; i < numOfEpochs; i++) {
			propagation.iteration();
		}
		
		propagation.finishTraining();
		
		return network;
		
	}
	
	public int getNdxBattle(){
		return NdxBattle;
	}

	private double [][] createSamples(int numInputs) {
		double [][] result =new double [NUMBATTLEFIELDSIZES*NUMCOOLINGRATES][numInputs];
		
		// Create samples
		for(int NdxBattleSize=0;NdxBattleSize<NUMBATTLEFIELDSIZES;NdxBattleSize++) {
			for(int NdxCooling=0;NdxCooling<NUMCOOLINGRATES;NdxCooling++) {
				result[NdxCooling+NdxBattleSize*NUMCOOLINGRATES][0]= 0.1+0.9*((double)NdxBattleSize)/NUMBATTLEFIELDSIZES;
				result[NdxCooling+NdxBattleSize*NUMCOOLINGRATES][1]= 0.1+0.9*((double)NdxCooling)/NUMCOOLINGRATES;
			} 
		}
		return result;
	}

	private void evaluateSamplesAndPlotImage(BasicNetwork network,double [][]MyTestData) {
		
		int[] OutputRGBint=new int[NUMBATTLEFIELDSIZES*NUMCOOLINGRATES]; 
		double MyValue=0;
		MLData testedInput;
		for(int NdxBattleSize=0;NdxBattleSize<NUMBATTLEFIELDSIZES;NdxBattleSize++) {
			for(int NdxCooling=0;NdxCooling<NUMCOOLINGRATES;NdxCooling++) {
				// Provide testing configuration as input to the network
				testedInput = new BasicMLData(MyTestData[NdxCooling + NdxBattleSize*NUMCOOLINGRATES]);
				// Collect output
				double MyResult= network.compute(testedInput).getData(0);
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
			e.printStackTrace();
		}
		System.out.println("Image generated.");
	}
	public void run() {
		runBattles();
		
		// Create the training dataset for the neural network
		final int NUM_NN_INPUTS = 2; // Adjust size based on number of observed parametres
		double [][]RawInputs=new double[NUMSAMPLES][NUM_NN_INPUTS]; 
		double [][]RawOutputs=new double[NUMSAMPLES][1];
		
		createDatasetForNN(RawInputs,RawOutputs);
		
		System.out.println("Training network...");
		
		BasicNetwork network = createAndTrainNN(RawInputs, RawOutputs, NUM_TRAINING_EPOCHS);
		
		System.out.println("Training completed.");
	
		
		System.out.println("Testing network...");
		
		evaluateSamplesAndPlotImage(network,createSamples(NUM_NN_INPUTS));
			
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
		BattlefieldParameterEvaluator evaluator;
		public BattleObserver(BattlefieldParameterEvaluator be) {
			evaluator = be;
		}
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
			Environment.FinalScore1[evaluator.getNdxBattle()] = results[0].getScore(); 
			Environment.FinalScore2[evaluator.getNdxBattle()] = results[1].getScore();
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
		engine.addBattleListener(new BattleObserver(this)); 
		
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
		
		// Show results
		System.out.println(Arrays.toString(BattlefieldSize)); 
		System.out.println(Arrays.toString(GunCoolingRate)); 
		System.out.println(Arrays.toString(FinalScore1)); 
		System.out.println(Arrays.toString(FinalScore2));
		
		// Cleanup our RobocodeEngine
		engine.close();
	}
}

