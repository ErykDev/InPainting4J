package org.inPainting.nn;


import lombok.Getter;
import lombok.Synchronized;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Random;

public final class ImageDataSetIterator implements MultiDataSetIterator {

    private Random r;
    private MultiDataSet[] multiDataSets;
    private MultiDataSetPreProcessor preProcessor = null;
    private int pointer = 0;

    private int iterationsPerPicture = 25;

    @Getter
    private int maxSize;

    public ImageDataSetIterator(int IterationsPerPicture, MultiDataSet[] multiDataSets){
        this.iterationsPerPicture = IterationsPerPicture;
        this.multiDataSets = multiDataSets;
        this.maxSize = (multiDataSets.length - 1) * IterationsPerPicture;
        this.r = new Random();

        this.shuffle();
    }

    public ImageDataSetIterator(MultiDataSet[] multiDataSets){
        this.multiDataSets = multiDataSets;
        this.maxSize = (multiDataSets.length - 1) * iterationsPerPicture;
        this.r = new Random();

        this.shuffle();
    }

    public ImageDataSetIterator(MultiDataSet[] multiDataSets, int seed){
        this.multiDataSets = multiDataSets;
        this.maxSize = (multiDataSets.length - 1) * iterationsPerPicture;
        this.r = new Random(seed);

        this.shuffle();
    }

    @Override
    @Synchronized
    public MultiDataSet next(int num) {
        return multiDataSets[num];
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        for (MultiDataSet multiDataSet : multiDataSets)
            preProcessor.preProcess(multiDataSet);
        this.preProcessor = preProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return this.preProcessor;
    }

    @Synchronized
    public MultiDataSet nextRandom(){
        return this.multiDataSets[this.r.nextInt(this.multiDataSets.length)];
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    @Synchronized
    public void reset() {
        this.pointer = 0;
        this.shuffle();
        System.gc();
    }

    @Synchronized
    public void shuffle() {
        MultiDataSet[] ar = this.multiDataSets;
        for (int i = ar.length - 1; i > 0; i--) {
            int index = r.nextInt(i + 1);
            MultiDataSet a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
        this.multiDataSets = ar;
    }

    @Override
    @Synchronized
    public boolean hasNext() {
        return (pointer < maxSize);
    }

    @Override
    @Synchronized
    public MultiDataSet next() {
        if (this.hasNext()){
            int dataPosition = (int)(pointer / iterationsPerPicture);
            pointer++;
            return multiDataSets[dataPosition];
        } else {
            return multiDataSets[multiDataSets.length-1];
        }
    }
}