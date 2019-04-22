function GPK = GP_Kernel(data1,data2,sigma_f,sigma_n,l)

    SE = @(x,y)(sigma_f*exp(-abs(x-y).^2/(2*l))+sigma_n*eq(x,y));
    GPK = bsxfun(SE, data1, data2);
end