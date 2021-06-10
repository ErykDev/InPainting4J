package org.inPainting.controller;

import javafx.application.Platform;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.concurrent.Task;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import lombok.SneakyThrows;
import lombok.Synchronized;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.inPainting.component.UIServerComponent;
import org.inPainting.nn.GAN;
import org.inPainting.nn.NeuralNetwork;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Component
@Slf4j
public class LearningGuiController {

    private final IntegerProperty counterProperty = new SimpleIntegerProperty();

    @Autowired
    UIServerComponent uiServerComponent;

    @FXML
    private Button btnLoad;

    @FXML
    private Button btnSave;

    @FXML
    private Button btnTest;

    @FXML
    private ToggleButton btnTrain;

    @FXML
    private Label counterText;

    @FXML
    private Label counterEpoch;

    @FXML
    private CheckBox TrainD;

    @Autowired
    private CustomLearningGuiController customLearningGuiController;

    private final File gan_file = new File("gan.zip");

    private final File disc_file = new File("discriminator.zip");

    private GAN gan;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    @FXML
    private void initialize() {
        tryToLoadNetworks();

        log.info("Discriminator");
        log.info(gan.getDiscriminator().summary());

        log.info("GAN");
        log.info(gan.getNetwork().summary());

        customLearningGuiController.onSetNeuralNetwork(gan);
        customLearningGuiController.onInitialize();

        uiServerComponent.reinitialize(gan.getNetwork());
        gan.setDiscriminatorListeners(new BaseTrainingListener[]{ new PerformanceListener(100, true) });

        counterProperty.addListener((observable, oldValue, newValue) -> {
            counterText.setText("Iteration: " + newValue);
            counterEpoch.setText("Epoch: " + ((newValue.longValue()/(customLearningGuiController).getDataSize())+1));
        });
    }

    public void loadAction(ActionEvent actionEvent) {
        Task<Void> loadTask = new Task<Void>() {
            @SneakyThrows
            @Override
            protected Void call() {
                btnLoad.setDisable(true);
                tryToLoadNetworks();

                customLearningGuiController.onSetNeuralNetwork(gan);
                customLearningGuiController.onInitialize();

                gan.setDiscriminatorListeners(new BaseTrainingListener[]{ new PerformanceListener(100, true) });
                return null;
            }
        };
        loadTask.setOnSucceeded(e -> {
            btnLoad.setDisable(false);
            showAlert(Alert.AlertType.INFORMATION, "Success", "Neural network successfully loaded");
        });

        executor.submit(loadTask);
    }

    @SneakyThrows
    private void tryToLoadNetworks(){
        if (gan_file.exists() && disc_file.exists()){
            try {
                gan =  new GAN(ComputationGraph.load(disc_file, true), ComputationGraph.load(gan_file, true));
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else
            gan = new GAN.Builder().discriminator(() -> {
                try {
                    log.info("Loading Discriminator");
                    return ComputationGraph.load(disc_file, true);
                } catch (IOException e) {
                    log.error("Error while loading discriminator network creating new one");
                    return NeuralNetwork.getDiscriminator();
                }
            }).updater(Adam.builder()
                    .learningRate(GAN.LEARNING_RATE)
                    .beta1(GAN.LEARNING_BETA1)
                    .build())
                    .build();
    }

    public void saveAction(ActionEvent actionEvent) {
        Task<Void> saveTask = new Task<Void>() {
            @SneakyThrows
            @Override
            protected Void call() {
                btnSave.setDisable(true);
                ModelSerializer.writeModel(gan.getNetwork(), gan_file, true);
                ModelSerializer.writeModel(gan.getDiscriminator(), disc_file,true);
                return null;
            }
        };
        saveTask.setOnSucceeded(e -> {
            btnSave.setDisable(false);
            showAlert(Alert.AlertType.INFORMATION, "Success", "Neural network successfully saved");
        });

        executor.submit(saveTask);
    }

    public void trainAction(ActionEvent actionEvent) {
        boolean trainingMode = btnTrain.isSelected();
        btnLoad.setDisable(trainingMode);
        btnSave.setDisable(trainingMode);
        btnTest.setDisable(trainingMode);
        if (btnTrain.isSelected())
            Platform.runLater(this::trainLoop);
    }

    public void testAction(ActionEvent actionEvent) {
        Task<Void> testTask = new Task<Void>() {
            @SneakyThrows
            @Override
            protected Void call() {
                btnTest.setDisable(true);
                customLearningGuiController.onTestAction();
                return null;
            }
        };

        testTask.setOnFailed(e -> {
            Throwable problem = testTask.getException();
            log.error("test error", problem);
        });

        testTask.setOnSucceeded(e -> btnTest.setDisable(false));

        executor.submit(testTask);
    }

    @Synchronized
    private void trainLoop() {
        Task<Void> executeAppTask = new Task<Void>() {
            @Override
            protected Void call() {
                customLearningGuiController.onTrainLoop(counterProperty.get(), TrainD.isSelected());
                Platform.runLater(() -> counterProperty.setValue(counterProperty.get() + 1));
                return null;
            }
        };
        executeAppTask.setOnSucceeded(e -> {
            if (btnTrain.isSelected()) {
                trainLoop();
            }
        });
        executeAppTask.setOnFailed(e -> {
            Throwable problem = executeAppTask.getException();
            log.error("learning loop error",problem);
        });

        executor.submit(executeAppTask);
    }

    private void showAlert(Alert.AlertType alertType, String title, String content) {
        Alert alert = new Alert(alertType);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(content);
        alert.showAndWait();
    }
}