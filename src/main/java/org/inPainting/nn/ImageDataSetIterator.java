package org.inPainting.nn;


import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Random;

public final class ImageDataSetIterator implements MultiDataSetIterator {

    private Random _r;
    private MultiDataSet[] _multiDataSets;
    private MultiDataSetPreProcessor _preProcessor = null;
    private int _pointer = 0;

    private int _iterationsPerPicture = 25;
    private int _maxSize;

    public ImageDataSetIterator(int IterationsPerPicture, MultiDataSet[] multiDataSets){
        this._iterationsPerPicture = IterationsPerPicture;
        this._multiDataSets = multiDataSets;
        this._maxSize = (multiDataSets.length - 1) * IterationsPerPicture;
        this._r = new Random();

        this.shuffle();
    }

    public ImageDataSetIterator(MultiDataSet[] multiDataSets){
        this._multiDataSets = multiDataSets;
        this._maxSize = (multiDataSets.length - 1) * _iterationsPerPicture;
        this._r = new Random();

        this.shuffle();
    }

    public ImageDataSetIterator(MultiDataSet[] multiDataSets, int seed){
        this._multiDataSets = multiDataSets;
        this._maxSize = (multiDataSets.length - 1) * _iterationsPerPicture;
        this._r = new Random(seed);

        this.shuffle();
    }

    @Override
    public synchronized MultiDataSet next(int num) {
        return _multiDataSets[num];
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        for (MultiDataSet multiDataSet : _multiDataSets)
            preProcessor.preProcess(multiDataSet);
        this._preProcessor = preProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return this._preProcessor;
    }

    public MultiDataSet nextRandom(){
        return this._multiDataSets[this._r.nextInt(this._multiDataSets.length)];
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
    public synchronized void reset() {
        this._pointer = 0;
        this.shuffle();
        System.gc();
    }

    public synchronized void shuffle() {
        MultiDataSet[] ar = this._multiDataSets;
        for (int i = ar.length - 1; i > 0; i--) {
            int index = _r.nextInt(i + 1);
            MultiDataSet a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
        this._multiDataSets = ar;
    }

    public long getSize(){
        return _maxSize;
    }

    @Override
    public synchronized boolean hasNext() {
        return (_pointer < _maxSize);
    }

    @Override
    public synchronized MultiDataSet next() {
        if (this.hasNext()){
            int dataPosition = (int)(_pointer/ _iterationsPerPicture);
            _pointer++;
            return _multiDataSets[dataPosition];
        } else {
            return _multiDataSets[_multiDataSets.length-1];
        }
    }
}