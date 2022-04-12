# ML-ttW
Machine Learning for ttW analysis

## Funcionamiento básico
Todo lo importante se hace en ```ttW_3l_training.py```. Si ejecutas:

``` python ttW_3l_training.py ``` 

Lo primero que va a hacer el código es usar la función ```load_data``` para
cargar variables que se encuentran almacenadas en nuestras rootfiles. Guardarse
estas variables tiene truco, pues hay algunas variables que están definidas
en lo que solemos llamar Friend-Trees.

Un Friend-Tree es esencialmente una rootfile que contiene información procesada
### Ejemplo: 
Llamamos rootfile principal a la rootfile a partir de la cuál se pueden generar
friend-trees. Estas rootfiles suelen tener información básica como el pT de los leptones,
variables cinemáticas, etc...

La variable de nJet25, que te dice el número de jets por suceso, es una
variable/branch que tenemos guardada en los Friend-trees de "1_recl_enero", pero
no la tenemos guardada en las rootfiles "principales".

Para cargar una variable proveniente de una rootfile principal usamos la lista
```vars_keep```, y para guardar variables de un friend-tree usamos otras listas
como por ejemplo la que viene definida en vars_friend_recl.

### Procesando los dataframes
Las variables que vienen en las rootfiles necesitan ser procesadas en cierto modo. Cuando
miramos a una rootfile, por ejemplo, el pT de los leptones no viene nunca separado por leptón,
sino que viene guardado en vectores ```[LepGood_pt[0], LepGood_pt[1], ..., LepGood_pt[nLepGood_pt]]```.
Es conveniente que a nuestro modelo de machine learning le pasemos información semi-procesada, es decir,
en vez de pasarle una variable con vectores de tres componentes, le pasemos tres columnas con una componente
cada una. Todo esto se hace en la función ``process_dataframes``. Ahí viene un ejemplo de cómo hacer esto
para el pT de los leptones.

### Aplicando cortes
Se pueden usar las variables almacenadas en el dataframe para quedarte con solo sucesos que cumplan un cierto
requisito. En el código de ejemplo viene cómo aplicar un corte que selecciona únicamente sucesos con 3 leptones y
con 2 o más jets.
