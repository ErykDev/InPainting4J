package org.inPainting.utils;

import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.inPainting.nn.data.ImageDataSetIterator;
import org.inPainting.nn.data.ImageFileDataSetIterator;
import org.inPainting.nn.data.ImageMemoryDataSetIterator;

import java.io.File;

public final class ImageLoader {

    private WritableImage writableTemp;

    public WritableImage emptyImage(Color color, int width, int height) {

        writableTemp = new WritableImage(width, height);
        PixelWriter pixelWriter = writableTemp.getPixelWriter();

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                pixelWriter.setColor(x, y, color);

        return writableTemp;
    }

    public WritableImage drawImage(INDArray data, int width, int height) {

        writableTemp = new WritableImage(width, height);
        PixelWriter pixelWriter = writableTemp.getPixelWriter();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double r = data.getDouble(0,0,y,x);
                double g = data.getDouble(0,1,y,x);
                double b = data.getDouble(0,2,y,x);
                Color color = new Color(r, g, b, 1);
                pixelWriter.setColor(x, y, color);
            }
        }
        return writableTemp;
    }



    public static INDArray mergeImagesByMask(INDArray IImage, INDArray Mask, INDArray OImage, int width, int height) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (Mask.getDouble(0,0,y,x) == 0.0D){
                    OImage.putScalar(new int[]{0,0,y,x},IImage.getDouble(0,0,y,x));
                    OImage.putScalar(new int[]{0,1,y,x},IImage.getDouble(0,1,y,x));
                    OImage.putScalar(new int[]{0,2,y,x},IImage.getDouble(0,2,y,x));
                }
            }
        }
        return OImage;
    }

    public ImageMemoryDataSetIterator prepareInMemoryData() {

        ImageDataSetIterator.FileEntry[] entries = new ImageFileDataSetIterator.FileEntry[new File(ImageLoader.class.getResource("/data/256/inputs/").getFile()).listFiles().length/2];

        for (int i = 1; i < entries.length + 1; i++) {
            entries[i-1] = new ImageDataSetIterator.FileEntry(
                    new File(ImageLoader.class.getResource("/data/256/inputs/input"+i+".png").getFile()),
                    new File(ImageLoader.class.getResource("/data/256/inputs/input"+i+"_mask.png").getFile()),
                    new File(ImageLoader.class.getResource("/data/256/expected/expected"+i+".png").getFile())
            );
        }
        return new ImageMemoryDataSetIterator(4, entries);
    }

    public ImageFileDataSetIterator prepareInFileData(){

        ImageFileDataSetIterator.FileEntry[] entries = new ImageFileDataSetIterator.FileEntry[new File(ImageLoader.class.getResource("/data/256/inputs/").getFile()).listFiles().length/2];
        for (int i = 1; i < entries.length + 1; i++) {
            entries[i-1] = new ImageFileDataSetIterator.FileEntry(
                    new File(ImageLoader.class.getResource("/data/256/inputs/input" +i+".png").getFile()),
                    new File(ImageLoader.class.getResource("/data/256/inputs/input"+i+"_mask.png").getFile()),
                    new File(ImageLoader.class.getResource("/data/256/expected/expected" +i+".png").getFile())
            );
        }
        return new ImageFileDataSetIterator(4, entries);
    }

    private static double scaleColor(double value) {
        return (value);
    }

    private static double scale(int value, int rangeSize) {
        return scale(value, rangeSize, 1.0d);
    }

    @SuppressWarnings("SameParameterValue")
    private static double scale(int value, int rangeSize, double targetRange) {
        return (targetRange / (double) rangeSize) * ((double) value) - targetRange * 0.5;
    }

    private static double trimToRange0to1(double value) {
        return trimToRange(value, 0.0, 1.0);
    }

    @SuppressWarnings("SameParameterValue")
    private static double trimToRange(double value, double min, double max) {
        return Math.max(Math.min(value, max), min);
    }
}