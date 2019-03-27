# Lane-detection

* Tools : python 3.6 + opencv

* Result

  ![Lane-detection-result](https://github.com/XinMiaoWang/Lane-detection/blob/master/result/result.gif)
  

一、介紹

  在駕駛輔助系統中，車道檢測有許多應用，例如:自動駕駛、車道偏離警示系統等等。在車道檢測技術中，雖然有很多方法被提出，但面對環境的各種變化(ex.光線、天氣、標線清晰程度)仍然難以克服。
  
  本文將用影像處利技術實作車道檢測系統，首先將彩色影像灰階化，方便後續的影像處理，再透過邊緣檢測技術(ex. Sobel、Canny)偵測出邊緣，並過濾非邊緣的點，最後利用霍夫轉換找出車道線。


二、研究方法

  2.1灰階影像
  
   彩色影像是由三原色組成，每一原色數值範圍在0~255之間，色彩數目共有16777216色(256x256x256)。每當系統運算時都需要處理大量的影像資料，這樣會造成運算速度變慢，也會占用大量的記憶體空間。所以將彩色影像轉換成灰階影像，可以降低資料量，以提高運算速度和減少記憶體空間的使用。
   
   灰階影像只有亮度變化，沒有色度資料，灰階代表從最暗到最亮之間不同亮度的層次級別。這中間層級越多，能夠呈現的畫面效果也越細膩。

   ![](https://i.imgur.com/raFMzCb.png)


  2.2去除雜訊
  
   高斯平滑化可以用來減少影像雜訊，相對於相鄰像素平均法會保留比較顯著的邊界對比特性。高斯平滑化會把每個像素值設為周圍相鄰像素值的加權平均，就是影像與高斯分布的遮罩做捲積運算，遮罩越大平滑效果越明顯。
    
  中值濾波與高斯平滑化不同的地方是，中值濾波不會做加減乘除的運算，而是將目前要處理的像素和它周邊的像素根據灰階數值的大小進行排序，再以排再最中間的灰階取代被處理像素的灰階。中值濾波的意義是強迫被處理像素的灰階儘量與周邊大多數像素的灰階相似，這樣可以保留邊界，但運算速度較慢，主要花在灰階的排序。
範例:

  ![](https://i.imgur.com/ukIUOVs.png)


  2.3邊緣檢測

  邊是在小區域中灰階變異最大的地方，所以幾乎都是用微分或差分來偵測邊，常用的微分有一次微分與二次微分。邊緣檢測的方法有很多種，根據微分次數大致可分為兩類，基於搜尋和基於零交叉。
  
  基於搜尋的邊緣檢測方法首先計算邊緣強度，通常用一次微分表示，例如梯度，然後用計算來估計邊緣的局部方向，通常採用梯度的方向，並利用此方向找到局部梯度的最大值。
  
  基於零交叉的方法找到由影像得到的二次微分的零交叉點來定位邊緣，通常用拉普拉斯。
  
   * 一次微分
   
      微分是有方向性的，將相鄰灰階在水平及垂直方向的微分向量，稱為梯度，因為邊緣檢測大都是利用灰階(亮度)的變化，所以這裡的梯度指的是灰階(亮度)梯度，使用這個特性就可以發現影像中可能是邊緣的地方，另外也可以利用差分(ex.遮罩捲積運算)取代微分來加速影像的處理。Prewitt、Sobel、Canny等方法就是利用梯度來偵測邊緣。

   -	梯度，表示影像f在(x, y)位置的一次微分

   ![](https://i.imgur.com/19hZ090.png)

   -	梯度強度定義

   ![](https://i.imgur.com/ZlFTR2o.png)

   -	方向定義

   ![](https://i.imgur.com/Jb2R5UK.png)

   -	準則定義

   ![](https://i.imgur.com/gu38LGJ.png)


  * Prewitt
  
    Prewitt是一種一次微分的邊緣檢測，利用像素點上下、左右鄰點間的梯度檢測邊緣，而在做差分運算前，會先用等比重的平均法來去除雜訊，減少做差分運算時被強化出來的雜訊，他的缺點是使用平均法去除雜訊容易將邊緣模糊掉。

      ![](https://i.imgur.com/b9ZUpN6.png)

  * Sobel
  
    Sobel是基於Prewitt的改進，不一樣的的方在於，他在做差分運算前，是用高斯分佈平均法去除雜訊，減少做差分運算時被強化出來的雜訊，而這個方法相對於Prewitt使用的等比重的平均法來說可以保留較好的邊緣特徵。

      ![](https://i.imgur.com/Nd25n7U.png)

  * Canny

    Canny演算法是由John Canny於1986年所開發出來的邊緣檢測演算法，其原理為利用像素之梯度的最大值來找邊緣點，演算法步驟如下:
    
    1、高斯平滑濾波器
    
      一張影像不可避免地都會包含或多或少數量的雜訊，大都要透過一些方法來降低雜訊，降低雜訊對邊緣檢測的影響，那麼經過高斯平滑濾波器後的灰階將變為:

      ![](https://i.imgur.com/3rBDbGh.png)

    2、計算梯度值和梯度方向

      ![](https://i.imgur.com/qn9o610.png)

      ![](https://i.imgur.com/ed3X7vM.png)

    3、非極大值抑制(non-maximum suppression)
    
      在前面的處理過程中，邊緣有可能被放大了(同一個邊緣被檢測出好幾個邊)，但我們不需要這麼多，只需要找到一個最能夠表達這個邊緣就夠了，而非極大值抑制可以過濾不是邊緣的點，將邊緣銳利化(變細)，使邊緣的寬度盡可能為1個piexl。如果一個像素屬於邊緣，那麼這個像素點在梯度方向上的梯度值會是最大的，不是邊緣就將灰度值設為0。

      ![](https://i.imgur.com/F9U91fv.png)

      根據John Canny提出的Canny演算法論文中，非極大值抑制只有在0、90、45、135這四個梯度方向上進行，每個像素點梯度方向按照相近程度用這四個方向來替代。

      這個情況下，非極大值抑制所比較的相鄰像素就是:
      
       ![](https://i.imgur.com/NFD8Aa2.png)


    4、雙閥值檢測邊緣
    
      經過非極大值抑制後，影像中仍然有許多雜訊，藉由閥值過濾可以減少雜訊的存在，而相較於只設定一個閥值，雙閥值的設定可以提高準確率，二個閥值分別為上界和下界。影像中的像素點如果大於閥值上界則被檢測為一定是邊界(強邊界)，小於閥值下界則被檢測為一定不是邊界，若是在兩者之間則被檢測為候選項(弱邊界)，對於這些像素點，如果他是與確定為邊緣的像素點相連，則判定為邊緣，否則為非邊緣。

      ![](https://i.imgur.com/A7Ha2aw.png)



  * 二次微分
  
    二次微分就是一次微分連續做二次，但是二次微分沒有方向性，而是代表純量的和，也就是兩個方向的二次微分和，是灰階(亮度)梯度的變化率。二次微分會在邊緣處產生一個極大值及一個極小值，稱為雙重邊，因此檢測其過零點可以得到梯度中的局部最大值。

  2.4 影像增強

  有時候單張影像偵測出來的邊緣太過薄弱，若直接對這種影像進行霍夫轉換，可能無法得到良好的結果，為了避免這個問題，我們可以在檢測車道線前，對薄弱的邊緣進行強化。
  
  我認為可以將目前的影像與前N張影像做OR運算，來增強車道線的特徵。

   ![](https://i.imgur.com/Yr1mVCy.png)

  2.5車道線檢測
  
  * 霍夫轉換
  
    霍夫轉換用在二值影像的形狀偵測，主要原理是利用影像中分散的點位置找出特定形狀(ex.直線或圓)的參數值。他的演算法流程大致如下，給定一個物件、要辨別的形狀的種類，演算法會在參數空間(parameter space)中執行投票來決定物體的形狀， 而這是由累加空間(accumulator space)裡的局部最大值(local maximum)來決定。
    
    以直線為例，將位置空間中共線的點對映到參數空間中會變成線，線代表產生參數的所有可能值，而這些線一定有共同交點，因此在參數空間中找較多線的交點(票數最多的交點)，就可以找到位置空間中的圖形參數。

    ![](https://i.imgur.com/2ZohOjg.png)

    使用 y = a x+b 代表一直線，在轉換到參數空間時會遇到一個問題，當一直線接近垂直時，斜率會趨近無限大，因此霍夫轉換要改用極座標(ρ, θ)表示。
  
    極座標與直角坐標轉換公式: 
    
    ![](https://i.imgur.com/wL35S2a.png)


    位置空間的點(x,y)轉換到參數空間時，會變成一條週期性曲線(ρ, θ)。如果將同一直線上的多個點轉換到參數空間，這些曲線會產生交點，找到票數最多的交點，即可找到位置空間中的圖形參數。

    ![](https://i.imgur.com/iVPGxRq.png)

三、實作過程

* 流程

  ![](https://i.imgur.com/67daEn9.png)

* 去除雜訊

  中值濾波會將濾波範圍內的所有像素排序，並用中值替換當前的像素值，在椒鹽噪音這種類型的雜訊，中值濾波能夠有效的去除雜訊，且模糊的現象比平均平滑和高斯平滑都來的輕微。

  ![](https://i.imgur.com/6Xp1Hnf.png)

* 邊緣檢測

  Canny演算法檢測輸入圖像的邊緣並在輸出圖像中標記出這些邊緣。
  
  Canny( const CvArr* image, CvArr* edges, double threshold1,double threshold2, int aperture_size=3 )

  ![](https://i.imgur.com/lHGmdWV.png)

  ![](https://i.imgur.com/52Q25ZW.png)

* 選取感興趣區域(Regions of Interest, ROI)

  這個方法是為了減少尋找影像中車道線的運算量，只要運算ROI範圍內的特徵即可，但是如果ROI選擇不正確會造成誤判或是搜尋不到車道線等情況，也可以去除掉車道以外的雜訊。
  
  輸入為Canny邊緣檢測的結果，三角形範圍是我們所選取的ROI(圖四)，然後可利用這個ROI產生一個mask (圖五)，這個mask是一個二值化的影像，黑色像素都是0，白色像素都是255 (圖六)。Canny邊緣檢測的結果也是一張二值化影像，所以只要將這張影像和mask做and運算(圖七)，就可以產生只有車道線的影像(圖八)。

  ![](https://i.imgur.com/YnraErT.png)

  ![](https://i.imgur.com/hKlWGOh.png)

  ![](https://i.imgur.com/oONKxeM.png)

  ![](https://i.imgur.com/8X4XaEv.png)

* 車道線偵測

  在OpenCV中霍夫直線偵測轉換有二種函數，分別為 HoughLines 和 HoughLinesP 來檢測圖像中的直線，HoughLines 是標準霍夫線變換和 HoughLinesP是統計概率霍夫線變換。函數 HoughLines 只能得到直線的參數 P、θ並不知道檢測到的直線的端點，所以會找出無窮長的直線，而HoughLinesP 是可以檢測到直線的兩個端點，所以找到的是一個線段。

  HoughLinesP(InputArray image, OutputArray lines, double rho, double theta, int threshold, double minLineLength=0, double maxLineGap=0 )

    -	rho:距離解析度，越小表示定位要求越準確，但也比較容易造成應該是同條線的點判為不同線。

    -	theta: 角度解析度，越小表示角度要求越準確，但也較易造成應該是同條線的點判為不同線。

    -	threshold: 累積個數閾值，大於這個閾值才會輸出相對應的直線。

    - minLineLength: 組成一條直線的最少點的數量，超過此值的線才會被保留，否則就丟掉。

    -	maxLineGap: 同一方向上二條線段判定為一條線段的最大允許間隔，值越大則允許線段之間斷裂的距離越大。
  

  * 霍夫轉換 HoughLinesP 函數直線變換方式：
  
      1、隨機抽取一個邊緣點，如果該點已經被標示為某一條直線上的點，則繼續在剩下的邊緣點中隨機抽取一個邊緣點，直到所有邊緣點都抽取完。

      2、對該點進行霍夫轉換，並進行累加和計算。

      3、選取參數空間中累積值最大的點，如果該點大於閥值，就進行步驟4，否則回到步驟1。

      4、根據霍夫轉換得到的最大值，從該點出發，沿著直線方向位移，進而找到直線的二個端點。

      5、計算直線的長度，如果符合條件，則保存此線段，並標記這個線段上的點不參與其他線段檢測的轉換。


      ![](https://i.imgur.com/I7jwMg0.png)


四、結果討論

  本文只實作了最基本的車道檢測方法，對於環境的變化不一定能夠良好的偵測出車道線，在ROI選取的部分，不同時間不同地點應選的ROI也跟著不同，而實作中的ROI是固定的，若能設計成動態選取ROI，更能適應環境變化，提高偵測準確度。
  
  在駕駛過程中有時候也會出現沒有車道線或是因為太多車造成車道線被遮擋等情況，我覺得可以使用Kalman Filter來預測下一個時間的車道線，也可以利用高斯混合模型的技術，因為同一車道線在前後幾個frame的影像中，位置變化緩慢，就是下一個frame的車道線位置與現在偵測出來的車道線位置不會相差太遠，根據這個特徵可以預測車道線位置，當相鄰二次檢測出來的車道線位置變化不大時，認為檢測結果正確，若位置變化很大，則丟掉當前檢測結果，使用歷史結果，歷史車道線用高斯混合模型建模。
