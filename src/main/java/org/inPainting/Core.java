package org.inPainting;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;

@SpringBootApplication
public class Core extends Application {
    private ConfigurableApplicationContext springContext;
    private Parent rootNode;
    private FXMLLoader fxmlLoader;
    public static void main(String[] args) {
        //CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);
        launch(args);
    }

    @Override
    public void init() {
        springContext = SpringApplication.run(Core.class);
        fxmlLoader = new FXMLLoader();
        fxmlLoader.setControllerFactory(springContext::getBean);
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        fxmlLoader.setLocation(getClass().getResource("/fxml/learning-gui.fxml"));

        primaryStage.setResizable(true);
        Scene scene = new Scene(fxmlLoader.load());
        primaryStage.setScene(scene);
        primaryStage.sizeToScene();
        primaryStage.setTitle("InPainting Training");
        primaryStage.setOnCloseRequest(event -> {
            stop();
            Platform.exit();
            System.exit(0);
        });
        primaryStage.show();
    }

    @Override
    public void stop() {
        springContext.stop();
    }
}