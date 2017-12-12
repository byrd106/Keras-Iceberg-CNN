import matplotlib.pyplot as plt


def savePlot(MetricType,history,name,e):

	if MetricType == "A":
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(str(e)+'_'+name+'_ACC_IMAGES.png')
		plt.clf()

	if MetricType == "L":
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(str(e)+'_'+name+'_LOSS_IMAGES.png')
		plt.clf()