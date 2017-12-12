
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def savePlot(MetricType,history,name,e):

	if MetricType == "A":
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		
		val = max(history.history['val_acc'])
		plt.plot([0, e-1], [val, val], 'k-', lw=2)

		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(str(e)+'_'+name+'_ACC_IMAGES.png')

		
		#hlines(0.5, 0, e)


		#plt.axvline(0.5, color='r')
		#axes = plt.gca()
		#axes.set_ylim([0.0,1.0])		
		plt.clf()

	if MetricType == "L":
		# axes = plt.gca()
		# axes.set_ylim([0.0,1.0])
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')

		val = min(history.history['val_loss'])
		plt.plot([0, e-1], [val, val], 'k-', lw=2)

		plt.legend(['train', 'test'], loc='upper left')		

		plt.savefig(str(e)+'_'+name+'_LOSS_IMAGES.png')
		plt.clf()
