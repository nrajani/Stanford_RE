package edu.stanford.nlp.kbp.slotfilling;


import edu.stanford.nlp.kbp.slotfilling.SlotfillingSystem;
//import edu.stanford.nlp.kbp.slotfilling.SlotfillingTasks;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.evaluate.SlotfillPostProcessor;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.Function;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

public class RunSlotfiller{

	public static void main(String[] args){
	    Properties props = new Properties();
	    InputStream input = null;
	    
	    try{
	    	input = new FileInputStream(args[0]);
	    	props.load(input);
	    }
	    catch(IOException ex){
	    	ex.printStackTrace();
	    }
	    
	    SlotfillingSystem.exec(new Function<Properties, Object>() {
	      @Override
	      public Object apply(Properties in) {
	    	  System.out.println("Success");
	       return null;
	      }
	    }, props);
	}
}
