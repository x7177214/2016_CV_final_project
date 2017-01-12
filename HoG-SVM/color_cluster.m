function color = color_cluster(im, colorCate, black, white, brown)




        d =zeros(1,3);
        
        weight = ones (1,256);
        weight(1, 110 :155 ) = 1.7;
        weight(1, 235 :256 ) = 0.3;
        weight(1, 1:16) = 0.7;
        
               
               im = imresize(im,[100,100]);
               
               im = imcrop(im , [20 20 60 60]);
               
               im = rgb2gray(im);
               im = im(:);
               im = single(im);
               h = hist(im,256);
               h = h / sum(h);
               
 
               d(1) = sum (((h - black).*weight).^2 ); %black
               d(2) = sum (((h - white).*weight).^2); %dwhite
               d(3) = sum (((h - brown).*weight).^2); %dBrown
               
               [~ , idx ] = min(d);
                color = colorCate{idx};
              