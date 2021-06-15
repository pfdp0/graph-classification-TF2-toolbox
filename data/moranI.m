function hatI = moranI(A, Y)
    B = double(A);
    Y = double(Y);
    %B(B > 0) = 1.0;
    %Asum = nnz(B);
    Asum = sum(B, 'all');
    N = size(B,1);
    
    hatI = 0;
    for k = 1:size(Y,2)
        y = Y(:,k);
        ym = mean(y);

        num = 0.0;
        den = 0.0;
        for i = 1:N
            den = den + ((y(i)-ym)^2);
            for j = 1:N
                num = num + (B(i,j)*(y(i)-ym)*(y(j)-ym));
            end
        end
        
        I = (N/Asum)*(num/den);
        hatI = hatI + I;
    
    hatI = hatI / size(Y,2);
    end
end
    
