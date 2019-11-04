import argparse
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dataloader
from utils import *
from network import *
from dataloader import UCF101_splitter
from opt_flow import opt_flow_infer
import timeit
import pyttsx3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    z='C:\\Users\\sumit\\Desktop\\cse_5\\Project\\Real-Time-Action-Recognition-master\\model_best.pth'
    model = Spatial_CNN(           
                        lr=5e-4,
                        resume='model_best.pth.tar',
                        demo=True
                        )
    model.run()


class Spatial_CNN():
    def __init__(self, lr,resume, demo):
        self.lr=lr
        self.resume=resume
        self.demo = demo

    def webcam_inference(self):

        frame_count = 0

        # config the transform to match the network's format
        transform = transforms.Compose([
                transforms.Resize((342, 256)),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # prepare the translation dictionary label-action
        data_handler = UCF101_splitter(os.getcwd()+'/UCF_list/', None)
        data_handler.get_action_index()
        class_to_idx = data_handler.action_label
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        print(idx_to_class)

        # Start looping on frames received from webcam
        vs = cv2.VideoCapture(0)
        softmax = torch.nn.Softmax()

        while True:
            # start = time.time()

            # read each frame and prepare it for feedforward in nn (resize and type)
            ret, orig_frame = vs.read()

            if ret is False:
                print ("Camera disconnected or not recognized by computer")
                break

            # if frame_count == 0:
            #     old_frame = orig_frame.copy()
            #
            # else:
            #     optical_flow = opt_flow_infer(old_frame, orig_frame)
            #     old_frame = orig_frame.copy()

            frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = transform(frame).view(1, 3, 224, 224).cpu()

            # feed the frame to the neural network

            # vote for class every 30 consecutive frames
            if frame_count % 10 == 0:
                nn_output = self.model(frame).cpu()
                nn_output = softmax(nn_output)
                nn_output = nn_output.data.cpu().numpy()
                preds = nn_output.argsort()[0][-5:][::-1]
                print(preds, "hello")
                pred_classes = [(idx_to_class[str(pred+1)], nn_output[0, pred]) for pred in preds]
                print(pred_classes[0][0])
                engine = pyttsx3.init()
                engine.say(pred_classes[0][0])
                engine.runAndWait()

            # Display the resulting frame and the classified action
            font = cv2.FONT_HERSHEY_SIMPLEX
            y0, dy = 300, 40
            for i in range(3):
                y = y0 + i * dy
                cv2.putText(orig_frame, '{} - {:.2f}'.format(pred_classes[i][0], pred_classes[i][1]),
                            (5, y), font, 1, (0, 0, 255), 2)
                #print('{} - {:.2f}'.format(pred_classes[i][0], pred_classes[i][1]))
            #print(pred_classes[0][0])
            k=pred_classes[0][0]
            cv2.imshow('Webcam', orig_frame)
            frame_count += 1
            # end = time.time()
            # print (end - start)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        vs.release()
        cv2.destroyAllWindows()


    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet101(pretrained= True, channel=3).cpu()
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cpu()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)

    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                map_location=torch.device('cpu')
                checkpoint = torch.load(self.resume,map_location=torch.device('cpu'))
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
           # else:
            #    print("==> no checkpoint found at '{}'".format(self.resume))
        #if self.evaluate:
         #   self.epoch = 0
          #  prec1, val_loss = self.validate_1epoch()
           # return

        if self.demo:
            print('1')
            self.model.eval()
            print('2')
            self.webcam_inference()

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True

        if self.demo:
            print("Entered")
            self.resume_and_evaluate()

        
    def frame2_video_level_accuracy(self):

        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds),101))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii=0
        for name in sorted(self.dic_video_level_preds.keys()):

            preds = self.dic_video_level_preds[name]
            label = int(self.test_video[name])-1

            video_level_preds[ii,:] = preds
            video_level_labels[ii] = label
            ii+=1
            if np.argmax(preds) == (label):
                correct+=1

        #top1 top5
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()

        top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))
        loss = self.criterion(Variable(video_level_preds).cpu(), Variable(video_level_labels).cpu())

        top1 = float(top1.numpy())
        top5 = float(top5.numpy())

        #print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
        return top1,top5,loss.data.cpu().numpy()



if __name__=='__main__':
    main()
