package org.inPainting.nn;


import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Random;

public class ImageDataSetIterator implements MultiDataSetIterator {

    private Random _r;
    private MultiDataSet[] _multiDataSets;
    private MultiDataSetPreProcessor _preProcessor = null;
    private int _pointer = 0;

    private int _eachPicture = 25;
    private int _maxSize;

    public ImageDataSetIterator(int IteratorPerPhoto, MultiDataSet[] multiDataSets){
        this._eachPicture = IteratorPerPhoto;
        this._multiDataSets = multiDataSets;
        this._maxSize = multiDataSets.length * _eachPicture;
        this._r = new Random();
    }

    public ImageDataSetIterator(MultiDataSet[] multiDataSets){
        this._multiDataSets = multiDataSets;
        this._maxSize = multiDataSets.length * _eachPicture;
        this._r = new Random();
    }

    public ImageDataSetIterator(MultiDataSet[] multiDataSets, int seed){
        this._multiDataSets = multiDataSets;
        this._maxSize = multiDataSets.length * _eachPicture;
        this._r = new Random(seed);
    }

    @Override
    public MultiDataSet next(int num) {
        return _multiDataSets[num];
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        for (int i = 0; i < _multiDataSets.length; i++)
            preProcessor.preProcess(_multiDataSets[i]);
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
    public void reset() {
        _pointer = 0;
    }

    @Override
    public boolean hasNext() {
        return (_pointer < _maxSize);
    }

    @Override
    public MultiDataSet next() {
        if (this.hasNext()){
            int dataPosition = (int)(_pointer/_eachPicture);
            _pointer++;
            return _multiDataSets[dataPosition];
        }else {
            this.reset();
            return _multiDataSets[0];
        }
    }
}
