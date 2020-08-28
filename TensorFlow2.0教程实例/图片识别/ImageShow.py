import matplotlib.pyplot as plt
class ImageShow:
    def __init__(self,images,labels,pred,index,num=5):
        self.images=images
        self.labels=labels
        self.pred=pred
        self.num=num
        self.index=index
        self.label_dict={}
    def showImage(self):
        fig=plt.gcf()
        fig.set_size_inches(12,6)
        if self.num>10:
            self.num=10
        for i in range(self.num):
            ax=plt.subplot(2,5,i+1)
            ax.imshow(self.images[i])
            title=str(i)+","+self.label_dict[self.labels[self.index][0]]
            if len(self.pred)>0:
                title +="=>"+self.label_dict[self.pred[self.index]]
            ax.set_title(title,fontsize=10)
            self.index+=1
        plt.show()

    def show_train_history(self,train_history, train, validation):
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title("Train History")
        plt.ylabel(train)
        plt.xlabel("Epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.show()

